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
import sys
import atexit
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
            #task_id = int(i*4//self.cfg.num_tasks)
            
            task_id = i%self.cfg.num_tasks

            task_type = self.task_types[task_id]
            generated_seq_len = int(normal[i])
            if generated_seq_len > 128:
                generated_seq_len = 128
            if generated_seq_len < 1:
                generated_seq_len = 1
            query_pool.append((task_id, generated_seq_len, task_type))

        self.query_pool = query_pool
    
    def prepare_tasks(self):
        print("Preparing PET Tasks")
        num_tasks = self.cfg.num_tasks
        random.seed(self.cfg.seed)
        for i in tqdm.tqdm(range(num_tasks)):
            pet_type = int(i//4)
            self.load_new_task(pet_type)


    def warmup(self):
        for i in range(20):
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
    
    def run(self, no_log = False, bs = 32 ):
        turbo_transformers.set_num_streams(self.cfg.num_streams)
        pet_scheduler = self.get_scheduler()
        batches = []
        for i in range(0, len(self.query_pool), bs):
            batch_queries = self.query_pool[i:i+bs]
            if len(batch_queries) > 0:
                task_ids = torch.LongTensor([q[0] for q in batch_queries])
                input_ids = torch.randint(low=0, 
                                          high=self.bert_cfg.vocab_size-1, 
                                          size=(len(batch_queries), self.cfg.mean_v), 
                                          dtype=torch.long, device=self.test_device)
                n_samples = torch.LongTensor([1 for _ in range(len(batch_queries))])

                batches.append((input_ids, task_ids, n_samples))

        self.warmup()      
        # Start serving------------:
        start = time.time()
        for iter in range(self.cfg.iterations):
            # for batch in tqdm.tqdm(batches):
            for batch in batches:
                # if len(batch) == 3:
                self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2])
                # elif len(batch) == 4:
                    # self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2], minibatch_lens = batch[3])
        
        elasp_time = time.time() - start
        average_time = elasp_time / self.cfg.iterations
        
        # print("Average time : {}".format(average_time),flush=True)
        QPS = self.cfg.total_queries / (average_time)
        print("QPS: {}".format(QPS),flush=True)

        if not no_log:
            self.write_log("QPS: {}\n".format(QPS))


    def compare_multi_stream(self):
        self.init_logger("multi_stream")
        # load up to 128 tasks
        self.cfg.schedule_policy = "batch_schedule"
        self.cfg.num_tasks = 32
        self.init()
        self.cfg.total_queries = 1024
        for task_num in [32]:
            self.cfg.num_tasks = task_num
            for i in range(1):
                for seq_len in [4,16,32,64,128,192,256]:
                    self.cfg.mean_v = seq_len
                    self.cfg.std_v = 0
                    self.prepare_query_pool()
                    for stream_num in [32,16,8,4,2,1]:
                        self.cfg.num_streams = stream_num
                        for b_size in [256,128,64,32]:
                            self.write_log("total_queries:{},task_num:{},stream_num:{},b_size:{},seq_len:{} ".format(self.cfg.total_queries, self.cfg.num_tasks, stream_num,b_size,seq_len))
                            self.run(bs = b_size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Which experiment to conduct?",
        default="batching_strategy",
        choices=["main_results","serving_throughput", "capacity_dense", "capacity_pet", "batching_strategy", "multi_stream", "breakdown"]
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
    else:
        raise NotImplementedError
 