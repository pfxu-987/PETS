import torch
# from torch._C import _load_for_lite_interpreter
from transformers.models.bert.modeling_bert import BertModel, BertConfig
import numpy
import turbo_transformers
import sys
import os
import time
import tqdm
from turbo_transformers.layers import PET_Types, PETBertModel, PETBertConfig

class Shadow_Server:
    def __init__(self, model_type = "bert_large") -> None:
        self.torch_model = None
        self.base_tt_model = None
        self.test_device = torch.device('cuda:0')

        self.model_type = model_type
        if model_type == 'distilbert':
            model_config = BertConfig(layers = 6)
        elif model_type == 'bert_large':
            model_config = BertConfig(layers=24, hidden_size = 1024, intermediate_size=4096, num_attention_heads=16)
        elif model_type == 'bert_base':
            model_config = BertConfig()
        self.cfg = model_config
    
    def load_torch_model(self):
        self.torch_model = BertModel(self.cfg)
        self.torch_model.eval()
        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

    def load_shared_w(self):
        base_turbo_model = turbo_transformers.SharedBertModel.from_torch(
            self.torch_model, self.test_device, "turbo")
        self.base_tt_model = base_turbo_model
        turbo_transformers.set_num_threads(4)
        del self.torch_model
    
    def log_simple_layer_shapes(self, model, log_file="model_layers_simple.log"):
        """
        只打印模型各层名称和形状，每个形状显示在层名的下一行
        """
        with open(log_file, "w") as f:
            f.write(f"模型类型: {self.model_type}\n\n")
            
            # 遍历所有命名参数获取层名和形状
            for name, param in model.named_parameters():
                # 显示所有参数，包括binary_mask
                f.write(f"{name}\n")
                f.write(f"{tuple(param.shape)}\n")
            
            # 添加总结信息
            f.write(f"\n总参数层数: {len(list(model.named_parameters()))}\n")
        
        print(f"模型层形状已记录到 {log_file}")
    
    def load_new_task(self, pet_type, model_path = None):
        """
        Load shadows 
        """
        if self.model_type == 'distilbert':
            pet_bert_config = PETBertConfig(layers = 6, pet_type = pet_type)
        elif self.model_type == 'bert_large':
            pet_bert_config = PETBertConfig(layers=24, hidden_size = 1024, intermediate_size=4096, num_attention_heads=16, pet_type = pet_type)
        elif self.model_type == 'bert_base':
            pet_bert_config = PETBertConfig(pet_type = pet_type)

        pet_bert_model = PETBertModel(pet_bert_config)
        if torch.cuda.is_available():
            pet_bert_model.to(self.test_device)
        mem_before_load = turbo_transformers.get_gpu_mem_usage()
        self.base_tt_model.load_new_task_from_torch(pet_bert_model)
        mem_after_load = turbo_transformers.get_gpu_mem_usage()
        self.memory_usage = mem_after_load - mem_before_load
        print("GPU memory usage for PET: {} MB".format(self.memory_usage))
        
        # 记录简化版的层形状信息
        pet_type_name = "unknown"
        if pet_type == PET_Types.adapters:
            pet_type_name = "adapters"
        elif pet_type == PET_Types.maskbert:
            pet_type_name = "maskbert"
        elif pet_type == PET_Types.diff_pruning:
            pet_type_name = "diff_pruning"
        elif pet_type == PET_Types.bitfit:
            pet_type_name = "bitfit"
        else:
            pet_type_name = f"type_{pet_type}"
            
        task_layers_simple_log_file = f"model_layers_{pet_type_name}.log"
        self.log_simple_layer_shapes(pet_bert_model, task_layers_simple_log_file)
        
        # 尝试获取模型中掩码相关的属性或缓冲区
        try:
            mask_log_file = f"model_masks_{pet_type_name}.log"
            with open(mask_log_file, "w") as f:
                f.write(f"模型类型: {self.model_type} - {pet_type_name}\n\n")
                
                # 检查是否有二进制掩码或其他特殊属性
                # 检查命名缓冲区（named_buffers）
                has_buffers = False
                for name, buffer in pet_bert_model.named_buffers():
                    has_buffers = True
                    f.write(f"Buffer: {name}\n")
                    f.write(f"{tuple(buffer.shape)}\n")
                    # 对于二进制掩码，显示非零元素的数量和百分比
                    if 'mask' in name:
                        non_zeros = torch.sum(buffer != 0).item()
                        total = buffer.numel()
                        percentage = 100.0 * non_zeros / total if total > 0 else 0
                        f.write(f"非零元素: {non_zeros}/{total} ({percentage:.2f}%)\n\n")
                    else:
                        f.write("\n")
                
                if not has_buffers:
                    f.write("没有找到命名缓冲区\n\n")
                
                # 检查是否有特殊属性
                special_attrs = ["binary_mask", "mask", "pruning_mask", "adapter"]
                found_special_attr = False
                
                for attr_name in special_attrs:
                    for name, module in pet_bert_model.named_modules():
                        if hasattr(module, attr_name):
                            found_special_attr = True
                            attr = getattr(module, attr_name)
                            f.write(f"特殊属性: {name}.{attr_name}\n")
                            if hasattr(attr, 'shape'):
                                f.write(f"{tuple(attr.shape)}\n\n")
                            else:
                                f.write(f"类型: {type(attr)}\n\n")
                
                if not found_special_attr:
                    f.write("没有找到特殊掩码属性\n")
            
            print(f"掩码信息已记录到 {mask_log_file}")
        except Exception as e:
            print(f"记录掩码信息时出错: {str(e)}")

    def init(self):
        self.load_torch_model()
        # 记录简化版的层形状信息
        self.log_simple_layer_shapes(self.torch_model, "model_layers_base.log")
        self.load_shared_w()

    def prepare_inputs(self, n_queries = 1024, bs_per_task = 1, seq_len = 128, n_tasks = 4):
        batches = []
        
        self.bs_per_task = bs_per_task
        #self.batch_size = 1
        self.seq_len = seq_len
        self.n_tasks = n_tasks

        self.batch_size = bs_per_task * n_tasks

        for i in range(n_queries // self.batch_size):
            task_ids = torch.arange(0, n_tasks).long()
            n_samples = torch.ones(n_tasks).long() * self.bs_per_task
            input_ids = torch.randint(low=0,
                                      high=self.cfg.vocab_size - 1,
                                      size=(self.batch_size, self.seq_len),
                                      dtype=torch.long,
                                      device=self.test_device)

            batch = [input_ids, task_ids, n_samples]
            batches.append(batch)

        return batches
        
    def run(self, batches, iterations = 100):
        # Warmup
        for i in range(5):
            for batch in batches:
                self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2])

        start = time.time()
        for i in range(iterations):
            for batch in batches:
                self.base_tt_model(batch[0], task_ids = batch[1], n_samples = batch[2])
        elasp_time = time.time() - start
        average_time = elasp_time / iterations
        QPS = self.batch_size * len(batches) / (average_time)
        print("Average time : {}".format(average_time))
        print("QPS : {}".format(QPS))
        with open("sequential_shadow_QPS_task.log", "a+") as f:
            f.write("{}, {}, {}, {}\n".format(self.n_tasks, self.bs_per_task, self.seq_len, QPS))

    def prepare_tasks(self, n_tasks):
        print("Loading {} tasks...".format(n_tasks))
        for i in tqdm.tqdm(range(n_tasks)):
           #server.load_new_task(PET_Types.adapters)
           #server.load_new_task(PET_Types.maskbert)
           # server.load_new_task(PET_Types.diff_pruning)
           server.load_new_task(PET_Types.bitfit)
        print("Done.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python " + sys.argv[0] + " n_tasks")
        quit()
    n_tasks = int(sys.argv[1])
    server = Shadow_Server()
    server.init()
    mem_before_load = turbo_transformers.get_gpu_mem_usage()
    server.prepare_tasks(n_tasks)
    mem_after_load = turbo_transformers.get_gpu_mem_usage()
    memory_usage = mem_after_load - mem_before_load
    print("GPU memory usage for weight: {} MB".format(memory_usage))
    
    for bs_per_task in [1]:
        for seq_len in [128]:
            #n_queries = 1024
            n_queries = bs_per_task * n_tasks
            inputs = server.prepare_inputs(n_queries, bs_per_task, seq_len, n_tasks)
            print("Start running for case {%d, %d, %d}..." % (n_tasks, bs_per_task, seq_len))
            server.run(inputs, iterations = 10)
            print("Stop running.")
