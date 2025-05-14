import torch
from torch.random import seed
from transformers.models.bert.modeling_bert import BertModel, BertConfig
import turbo_transformers
import time
from turbo_transformers.layers import PET_Types, PETBertModel, PETBertConfig
import random 
import argparse
import tqdm
import numpy as np
import os
from pet_scheduler import PET_Scheduler

class PET_Server:
    def __init__(self,cfg) -> None:
        self.torch_model = None
        self.base_tt_model = None
        self.test_device = torch.device('cuda:0')
        self.cfg = cfg  # experiment config
        self.logger = None        
        # init the bert model config
        if cfg.model == 'distilbert':
            model_config = BertConfig(num_hidden_layers = 6)
        elif cfg.model == 'bert_large':
            model_config = BertConfig(num_hidden_layers = 24, hidden_size = 1024,
                                      intermediate_size=4096,
                                      num_attention_heads=16)
        elif cfg.model == 'bert_base':
            model_config = BertConfig()
        else:
            raise NotImplementedError
        self.bert_cfg = model_config

        self.task_types = []
        # self.init_logger()

        self.has_inited = False

    def init_logger(self,exp_name):
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        log_name = os.path.join(self.cfg.log_dir,exp_name+"_PETS.log")
        self.logger = open(log_name,"w")

    def write_log(self, str):
        self.logger.write(str)
        self.logger.flush()
        
    def load_torch_model(self):
        self.torch_model = BertModel(self.bert_cfg)
        self.torch_model.eval()
        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

    def load_shared_weight(self):
        """
        Load the pytorch model weight as the shared parameters
        """
        base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.torch_model, self.test_device, "turbo")
        self.base_tt_model = base_turbo_model
        turbo_transformers.set_num_threads(4)

        # release the torch model
        self.torch_model = None
        torch.cuda.empty_cache()

    def load_dense_model(self):
        base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.torch_model, self.test_device, "turbo")
        return base_turbo_model

    def load_new_task(self, pet_type, model_path = None):
        """
        Load PETs
        """
        if self.cfg.model == 'distilbert':
            pet_bert_config = PETBertConfig(num_hidden_layers = 6, pet_type = pet_type)
        elif self.cfg.model == 'bert_large':
            pet_bert_config = PETBertConfig(num_hidden_layers=24, hidden_size = 1024,
                                            intermediate_size=4096,
                                            num_attention_heads=16,
                                            pet_type = pet_type)
        elif self.cfg.model == 'bert_base':
            pet_bert_config = PETBertConfig(pet_type = pet_type)

        pet_bert_model = PETBertModel(pet_bert_config)

        if torch.cuda.is_available():
            pet_bert_model.to(self.test_device)
        
        self.task_types.append(pet_type)
        self.base_tt_model.load_new_task_from_torch(pet_bert_model)

    def init(self):

        if self.has_inited:
            # if run the all-in-one script
            return 

        # shared model
        self.load_torch_model()
        self.load_shared_weight()
        print("Shared model loaded")

        # PET tasks
        self.prepare_tasks()
        print("PET tasks loaded")

        self.has_inited = True
        # Queries 
        # self.prepare_query_pool()
        # print("Benchmark generated")

    def configure_task_stream_mapping(self, mapping_dict=None):
        """
        Configure custom mapping between task IDs and stream IDs
        
        Args:
            mapping_dict: Dictionary mapping task_id -> stream_id
                         If None, clear all mappings and use default (task_id % num_streams)

        server = PET_Server(cfg)
        server.init()

        # 1. 通过字典设置手动映射    
        manual_mapping = {
            0: 3,  # 任务0映射到流3
            1: 2,  # 任务1映射到流2
            2: 1,  # 任务2映射到流1
            3: 0,  # 任务3映射到流0
            # ... 其他任务映射
        }

        server.configure_task_stream_mapping(manual_mapping)

        # 运行实验
        server.run(custom_stream_mapping=manual_mapping)

        # 2. 或者直接在run方法中提供映射
        custom_mapping = {i: (i*2) % server.cfg.num_streams for i in range(server.cfg.num_tasks)}
        server.run(custom_stream_mapping=custom_mapping)

        """
        import turbo_transformers
        
        if mapping_dict is None:
            # Clear any existing mappings, revert to default behavior
            turbo_transformers.clear_task_to_stream_mapping()
            print("Task-to-stream mapping cleared. Using default mapping (task_id % num_streams)")
            return
        
        # Set custom mappings for each task_id
        for task_id, stream_id in mapping_dict.items():
            if stream_id >= self.cfg.num_streams:
                print(f"Warning: Stream ID {stream_id} is out of range (max: {self.cfg.num_streams-1}). "
                      f"Will be mapped to {stream_id % self.cfg.num_streams}")
            turbo_transformers.set_task_to_stream_mapping(task_id, stream_id)
        
        print(f"Set custom task-to-stream mapping for {len(mapping_dict)} tasks")
    
    def create_stream_mapping(self, strategy="default", **kwargs):
        """

        # 设置总的流数量
        turbo_transformers.set_num_streams(4)  # 例如设置4个流

        # 手动映射 - 直接设置单个映射
        turbo_transformers.set_task_to_stream_mapping(0, 2)  # 将任务0映射到流2
        turbo_transformers.set_task_to_stream_mapping(1, 1)  # 将任务1映射到流1
        turbo_transformers.set_task_to_stream_mapping(2, 3)  # 将任务2映射到流3

        # 清除所有映射，恢复默认的映射方式 (task_id % num_streams)
        turbo_transformers.clear_task_to_stream_mapping()

        """
        if strategy == "default":
            # Default mapping is handled by the system, return None
            return None
            
        elif strategy == "workload_aware":
            # Distribute intensive tasks, group others
            intensive_tasks = kwargs.get("intensive_tasks", [0, 1, 2, 3])
            
            mapping = {}
            # Spread intensive tasks across streams
            for i, task_id in enumerate(intensive_tasks):
                mapping[task_id] = i % self.cfg.num_streams
                
            # Group remaining tasks
            remaining_tasks = [i for i in range(self.cfg.num_tasks) 
                              if i not in intensive_tasks]
            group_size = kwargs.get("group_size", 3)
            
            for i, task_id in enumerate(remaining_tasks):
                # Skip stream 0 if possible to reduce contention with intensive tasks
                offset = 1 if self.cfg.num_streams > 1 else 0
                stream_id = (offset + (i // group_size)) % self.cfg.num_streams
                mapping[task_id] = stream_id
                
            return mapping
        
        else:
            raise ValueError(f"Unknown stream mapping strategy: {strategy}")
    
    def prepare_query_pool(self):

        """
        Generate queries obeying normal distribution 
        """
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        query_pool = []
        normal = np.random.normal(self.cfg.mean_v, self.cfg.std_v, self.cfg.total_queries)
    
        for i in range(self.cfg.total_queries):
            # randomly assign the query to a task
            task_id = i%self.cfg.num_tasks

            task_type = self.task_types[task_id]
            generated_seq_len = int(normal[i])
            if generated_seq_len > 256:
                generated_seq_len = 256
            if generated_seq_len < 1:
                generated_seq_len = 1
            query_pool.append((task_id, generated_seq_len, task_type))

        self.query_pool = query_pool
    
    def prepare_tasks(self):
        print("Preparing PET Tasks")
        num_tasks = self.cfg.num_tasks
        random.seed(self.cfg.seed)
        for _ in tqdm.tqdm(range(num_tasks)):
            pet_type = 3
            self.load_new_task(pet_type)

    def warmup(self):
        for i in range(10):
            input_ids = torch.randint(low=0,
                                  high=self.bert_cfg.vocab_size - 1,
                                  size=(4, 128),
                                  dtype=torch.long,
                                  device=self.test_device)
            task_ids = torch.LongTensor([0,1,2,3])
            # task_ids = torch.LongTensor([0])
            n_samples = torch.LongTensor([1,1,1,1])
            self.base_tt_model(input_ids, task_ids = task_ids, n_samples = n_samples)
    
    def get_scheduler(self):
        # schedule the quiery pool to get batches
        pet_scheduler = PET_Scheduler(query_pool=self.query_pool,
                                      vocab_size=self.bert_cfg.vocab_size,
                                      sort_queries=self.cfg.sort_queries,
                                      test_device=self.test_device,
                                      alpha_table_path = self.cfg.alpha_table_path,
                                      beta_table_path = self.cfg.beta_table_path
                                      )
        return pet_scheduler
    
    def run(self, no_log=False, bs=32, custom_stream_mapping=None):

        turbo_transformers.set_num_streams(self.cfg.num_streams)
        
        pet_scheduler = self.get_scheduler()
        # Schedule the queries
        if self.cfg.schedule_policy == "batch_schedule":
            batches = pet_scheduler.batch_schedule(bs)
        elif self.cfg.schedule_policy == "two_stage_schedule":
            batches = pet_scheduler.coordinate_schedule(stage = self.cfg.schedule_stage)
        # Warmup
        self.warmup()      
        # Start serving------------:
        start = time.time()
        for iter in range(self.cfg.iterations):
            # for batch in tqdm.tqdm(batches):
            for batch in batches:
                if len(batch) == 3:
                    self.base_tt_model(batch[0], task_ids=batch[1], n_samples=batch[2])
                elif len(batch) == 4:
                    self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2], minibatch_lens = batch[3])
        
        elasp_time = time.time() - start
        average_time = elasp_time / self.cfg.iterations
        
        # print("Average time : {}".format(average_time),flush=True)
        QPS = self.cfg.total_queries / (average_time)
        print("QPS: {}".format(QPS),flush=True)

        if not no_log:
            self.write_log("QPS: {}\n".format(QPS))
            
        # Clear custom mapping after run
        if custom_stream_mapping is not None:
            self.configure_task_stream_mapping(None)

    def compare_batching(self):
        self.init_logger("compare_batching")
        # load up to 128 tasks
        self.cfg.num_tasks = 32
        self.cfg.num_streams = 1
        self.init()
        self.cfg.total_queries = 1024
        for task_num in [32]:
            for mean_v in [32,64,128]:
                for std_v in [1,4]:
                    self.cfg.num_tasks = task_num
                    self.cfg.mean_v = mean_v
                    self.cfg.std_v = std_v
                    self.prepare_query_pool()
                    
                    # fixed bactch
                    self.cfg.schedule_policy = "batch_schedule"
                    for num_streams in [1,2,4,8,16]:
                        self.cfg.num_streams = num_streams
                        cur_cfg = "total_queries:{},task_num:{},mean_v:{},std_v:{},stage:{},num_streams:{} ".format(self.cfg.total_queries, self.cfg.num_tasks, mean_v, std_v, 0, num_streams)
                        print(cur_cfg,flush=True)
                        self.write_log(cur_cfg)
                        self.run(bs = 128)

                    # batch scheduling 
                    self.cfg.schedule_policy = "two_stage_schedule"
                    for stage in [1,2,3,4]:
                        self.cfg.schedule_stage = stage
                        for num_streams in [1,2,4,8,16,32]:
                            self.cfg.num_streams = num_streams
                            cur_cfg = "total_queries:{},task_num:{},mean_v:{},std_v:{},stage:{},num_streams:{} ".format(self.cfg.total_queries, self.cfg.num_tasks, mean_v, std_v, stage, num_streams)
                            print(cur_cfg,flush=True)
                            self.write_log(cur_cfg)
                            self.run(bs = 128)

    def compare_multi_stream(self):
        self.init_logger("multi_stream")
        # load up to 128 tasks
        self.cfg.schedule_policy = "batch_schedule"
        self.cfg.num_tasks = 32
        self.init()
        self.cfg.total_queries = 1024
        for task_num in [32]:
            self.cfg.num_tasks = task_num
            for seq_len in [4, 32, 64,128]:
                self.cfg.mean_v = seq_len
                self.cfg.std_v = 0
                self.prepare_query_pool()
                for stream_num in [32, 16, 8, 4, 2, 1]:
                    self.cfg.num_streams = stream_num
                    self.write_log("total_queries:{},task_num:{},stream_num:{},seq_len:{} ".format(self.cfg.total_queries, self.cfg.num_tasks, stream_num,seq_len))
                    self.run(bs = 128)

    def explore_capacity_pet(self):
        self.cfg.schedule_policy = "batch_schedule"
        # self.cfg.num_tasks = 70
        self.init()
        
        self.cfg.total_queries = 2048
        seq_len = 128
        self.cfg.mean_v = seq_len
        self.cfg.std_v = 0
        self.prepare_query_pool()
        step = 8
        while(True):
            for _ in range(step):
                pet_type = random.randint(0, 3)
                self.load_new_task(pet_type)
                self.cfg.num_tasks += 1
            self.run(no_log = True)
            print("task_num:{}".format(self.cfg.num_tasks),flush=True)

    def explore_capacity_dense(self):
        """
        Evaluate the maximum number of supported dense models
        """
        self.load_torch_model()
        N = 1
        dense_models = []
        while(True):
            dense_models.append(self.load_dense_model())
            N += 1
            print(N)

    def serving_throughput(self):
        self.init_logger("serving_throughput")
        self.cfg.schedule_policy = "batch_schedule"
        # load a max number of tasks 
        self.cfg.num_tasks = 64
        self.cfg.num_streams = 1
        #self.cfg.num_tasks = 16
        self.init()
        
        for bs, seq_len in [(1,128), (1,64), (2,64), (2,32),(4,32), (4,16)]:
            for task_num in [64, 32, 16, 8, 4, 2, 1]:
            #for task_num in [16, 8, 4, 2, 1]:
                self.cfg.num_tasks = task_num
                self.cfg.total_queries = bs * task_num
                self.cfg.mean_v = seq_len
                self.cfg.std_v = 0
                self.prepare_query_pool()
                cur_cfg = "task_num:{},bs:{},seq_len:{},stream_num:{} ".format(task_num, bs, seq_len, self.cfg.num_streams)
                print(cur_cfg,flush=True)
                self.write_log(cur_cfg)
                self.run(bs = len(self.query_pool))

    def breakdown(self):
        # self.init_logger("breakdown")
        self.cfg.schedule_policy = "batch_schedule"
        self.cfg.num_tasks = 8
        self.init()
        for bs, seq_len in [(1,64), (2,32)]:
            self.cfg.mean_v = seq_len
            self.cfg.std_v = 0
            self.cfg.total_queries = bs * self.cfg.num_tasks
            self.prepare_query_pool()
            for stream_num in [1]:
                self.cfg.num_streams = stream_num
                print("bs:{},seq_len:{},task_num:{},stream_num:{} ".format(bs, seq_len, self.cfg.num_tasks, stream_num),flush=True)
                turbo_transformers.enable_perf("PETS")
                self.run(no_log = True, bs = len(self.query_pool))
                turbo_transformers.print_results()
                
    def custom_stream_mapping_demo(self):
        """Demonstrate the use of custom task-to-stream mapping"""
        self.init_logger("custom_stream_mapping")
        self.cfg.schedule_policy = "batch_schedule"
        self.cfg.num_tasks = 16
        self.cfg.num_streams = 1
        self.init()
        
        # Test parameters
        bs = 8
        seq_len = 4
        self.cfg.mean_v = seq_len
        self.cfg.std_v = 0
        self.cfg.total_queries = bs * self.cfg.num_tasks
        self.prepare_query_pool()
        
        # Helper function to print task to stream mappings
        def print_task_stream_mappings():
            import turbo_transformers
            print("\nCurrent task to stream mappings:")
            for task_id in range(self.cfg.num_tasks):
                stream_id = turbo_transformers.get_stream_id_for_task(task_id)
                print(f"  Task {task_id} -> Stream {stream_id}")
            print()
        
        # First, run with default mapping (task_id % num_streams)
        print("\n" + "="*80)
        print("EXPERIMENT 1: Default mapping (task_id % num_streams)")
        print("="*80)
        self.configure_task_stream_mapping(self.create_stream_mapping("default"))
        print_task_stream_mappings()
        self.run(no_log=False, bs=len(self.query_pool))
        
        # Now create some custom mappings for demonstration
        # Mapping schema 1: Group tasks in 4's to a stream
        print("\n" + "="*80)
        print("EXPERIMENT 2: Grouped mapping (4 tasks per stream)")
        print("="*80)
        grouped_mapping = self.create_stream_mapping("group_by", group_size=4)
        self.configure_task_stream_mapping(grouped_mapping)
        print_task_stream_mappings()
        self.run(no_log=False, bs=len(self.query_pool), custom_stream_mapping=grouped_mapping)
        
        # Mapping schema 2: Round-robin distribution across streams
        print("\n" + "="*80)
        print("EXPERIMENT 3: Round-robin distribution")
        print("="*80)
        round_robin_mapping = self.create_stream_mapping("round_robin")
        self.configure_task_stream_mapping(round_robin_mapping)
        print_task_stream_mappings()
        self.run(no_log=False, bs=len(self.query_pool), custom_stream_mapping=round_robin_mapping)
        
        # Mapping schema 3: All tasks to stream 0 (essentially single-stream)
        print("\n" + "="*80)
        print("EXPERIMENT 4: Single-stream mapping (all to stream 0)")
        print("="*80)
        single_stream_mapping = self.create_stream_mapping("single_stream", stream_id=0)
        self.configure_task_stream_mapping(single_stream_mapping)
        print_task_stream_mappings()
        self.run(no_log=False, bs=len(self.query_pool), custom_stream_mapping=single_stream_mapping)
        
        # For direct comparison with a true single-stream setup
        print("\n" + "="*80)
        print("EXPERIMENT 5: True single-stream (num_streams=1)")
        print("="*80)
        # Save current num_streams value
        original_streams = self.cfg.num_streams
        # Set to single stream
        self.cfg.num_streams = 1
        self.configure_task_stream_mapping(None)
        print_task_stream_mappings()
        self.run(no_log=False, bs=len(self.query_pool))
        # Restore original num_streams
        self.cfg.num_streams = original_streams
        
        # Advanced scenario: Workload-aware stream allocation
        print("\n" + "="*80)
        print("EXPERIMENT 6: Workload-aware mapping")
        print("Scenario: Tasks 0-3 are compute-intensive, spread across streams")
        print("          Tasks 4-15 are less intensive, grouped by stream")
        print("="*80)
        
        # Define tasks 0-3 as compute-intensive
        compute_intensive_tasks = [0, 1, 2, 3]
        workload_mapping = self.create_stream_mapping(
            "workload_aware", 
            intensive_tasks=compute_intensive_tasks,
            group_size=3
        )
        
        self.configure_task_stream_mapping(workload_mapping)
        print_task_stream_mappings()
        self.run(no_log=False, bs=len(self.query_pool), custom_stream_mapping=workload_mapping)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Which experiment to conduct?",
        default="batching_strategy",
        choices=["main_results","serving_throughput", "capacity_dense", "capacity_pet", "batching_strategy", "multi_stream", "breakdown", "custom_stream_mapping"]
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        # required=True,
        help="Path to write the results",
        default="./"
    )

    parser.add_argument(
        "--num_tasks",
        type=int,
        default = 128,
        help="Number of loaded tasks",
    )
    parser.add_argument(
        "--test_device",
        type=str,
        default = 'cuda:0',
    )
    parser.add_argument(
        "--min_seq_length",
        type=int,
        default = 8,
    )
    parser.add_argument(
        "--max_seq_length",
        type = int,
        default = 64
    )
    parser.add_argument(
        "--seed",
        type=int,
        default = 1
    )
    parser.add_argument(
        "--model",
        type = str,
        default = "bert_base",
        choices=['bert_base', 'distilbert', 'bert_large']
    )
    parser.add_argument(
        "--total_queries",
        type = int, 
        default = 1024,
        help = "Total number of queries in the pool"
    )
    parser.add_argument(
        "--iterations",
        type = int, 
        default = 10,
        help = "Total number of iterations"
    )
    parser.add_argument(
        "--sort_queries",
        action = "store_true"
    )
    parser.add_argument(
        "--num_streams",
        type = int, 
        default = 1,
        help = "Total number of CUDA streams"
    )

    parser.add_argument(
        "--alpha_table_path",
        type = str,
        default = "perf_model/alpha_table_1080ti_256_128_4.dat",
    )
    parser.add_argument(
        "--beta_table_path",
        type = str,
        default = "perf_model/beta_table_1080ti.dat",
    )
    parser.add_argument(
        "--schedule_policy",
        type=str,
        default = "batch_schedule",
        choices=["batch_schedule","two_stage_schedule"]
    )
    
    parser.add_argument(
        "--schedule_stage",
        type=int,
        default = 2
    )
    parser.add_argument(
        "--mean_v",
        type=int,
        default = 32
    )
    parser.add_argument(
        "--std_v",
        type=int,
        default = 1
    )

    cfg = parser.parse_args()
    server = PET_Server(cfg)

    # conduct an experiment
    if cfg.exp_name == "main_results":
        server.cfg.num_tasks = 128
        server.init()
        server.serving_throughput()
        server.compare_batching()
        server.compare_multi_stream()

    elif cfg.exp_name == "serving_throughput":
        server.serving_throughput()
    elif cfg.exp_name == "batching_strategy":
        server.compare_batching()
    elif cfg.exp_name == "multi_stream":
        server.compare_multi_stream()
    
    elif cfg.exp_name == "capacity_dense":
        server.explore_capacity_dense()
    elif cfg.exp_name == "capacity_pet":
        server.explore_capacity_pet()
    elif cfg.exp_name == "breakdown":
        server.breakdown()
    elif cfg.exp_name == "custom_stream_mapping":
        server.custom_stream_mapping_demo()
    else:
        raise NotImplementedError
