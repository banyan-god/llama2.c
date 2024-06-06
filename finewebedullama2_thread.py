import threading
from collections import deque
from datasets import load_dataset
import torch
import transformers 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import time
from datasets.distributed import split_dataset_by_node


class Task:
    def __init__(self,batch_size,device,block_size, prefetch_buffer=50):
        self.device=device
        self.block_size = block_size
        self.batch_size=batch_size

        self.prefetch_buffer = prefetch_buffer
        self.train_batch_queue = deque()
        self.val_batch_queue = deque()
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = self.tokenizer.vocab_size

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')  # Adjust as per your setup

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.initialized = False
        self.initialize()
        
        self.prefetch_train_thread = threading.Thread(target=self.prefetch_batches, args=("train",))
        self.prefetch_val_thread = threading.Thread(target=self.prefetch_batches, args=("val",))
        self.prefetch_train_thread.daemon = True
        self.prefetch_val_thread.daemon = True
        self.prefetch_train_thread.start()
        self.prefetch_val_thread.start()
        
        
    def initialize(self):
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", split="train", streaming=False,num_proc=24)
        iterable_dataset = dataset.to_iterable_dataset()
        shuffled_dataset = iterable_dataset.shuffle(buffer_size=100000,seed=42)
          # Skip the first 1000 records for validation
        val_dataset_r = shuffled_dataset.take(10000) 
        train_dataset_r = shuffled_dataset.skip(10000)
        val_dataset = split_dataset_by_node(val_dataset_r, rank=self.rank, world_size=self.world_size)
        train_dataset = split_dataset_by_node(train_dataset_r, rank=self.rank, world_size=self.world_size)
        tokenized_datasets = train_dataset.map(self.tokenize_function, batched=True, remove_columns={'id','url','text','dump','file_path','language','language_score','token_count','score','int_score'})
        tokenized_val_datasets = val_dataset.map(self.tokenize_function, batched=True,remove_columns={'id','url','text','dump','file_path','language','language_score','token_count','score','int_score'})


        
        # tokenized_datasets = tokenized_datasets.with_format("torch")
        # tokenized_val_datasets=tokenized_val_datasets.with_format("torch")
        self.train_loader = DataLoader(tokenized_datasets, batch_size=self.batch_size,   pin_memory=True)
        self.val_loader = DataLoader(tokenized_val_datasets, batch_size=self.batch_size,  pin_memory=True)
        self.initialized = True
        
    def tokenize_function(self,examples):
        # Tokenize the text
        tokenized_output = self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=self.block_size, return_tensors='pt')
        tokenized_output['input_ids'] = tokenized_output['input_ids'].long()
    
    
        # Shift input_ids to create labels and append -100
        labels = torch.cat([tokenized_output['input_ids'][:, 1:], torch.full((tokenized_output['input_ids'].shape[0], 1), -100)], dim=1)
        
        # Add labels to the tokenized output
        tokenized_output['labels'] = labels
        return tokenized_output
        
    def prefetch_batches(self, mode):
        try:
            if mode == "train":
                iterator = iter(self.train_loader)
                queue = self.train_batch_queue
            elif mode == "val":
                iterator = iter(self.val_loader)
                queue = self.val_batch_queue
            else:
                raise ValueError("Mode must be 'train' or 'val'.")

            while True:
                if len(queue) < self.prefetch_buffer:
                    batch = next(iterator, None)
                    if batch is not None:
                        queue.append(batch)
                    else:
                        print("Batch is none for the split", mode)
                else:
                    time.sleep(0.1) 

        except Exception as e:
            print(f"Exception in prefetching {mode} batches: {e}")


                
    def process_batch(self, batch):
        X = batch['input_ids'].detach().to(self.device, non_blocking=True)
        Y = batch['labels'].detach().to(self.device, non_blocking=True)
        return X, Y
        
    def next_batch(self, mode):
        if mode == "train" and self.train_batch_queue:
            return self.process_batch(self.train_batch_queue.popleft())
        elif mode == "val" and self.val_batch_queue:
            return self.process_batch(self.val_batch_queue.popleft())
        else:
            return None
            
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     task = Task(10, device, 1024)
#     train_batch_count = 0
#     val_batch_count = 0
#     total_batches = 50  # Set total number of batches for train and val


#     # Main training loop
#     try:
#         while train_batch_count < total_batches or val_batch_count < total_batches:
#             if train_batch_count < total_batches:
#                 # Fetch a batch for training
#                 train_batch = task.next_batch("train")
#                 if train_batch:
#                     train_X, train_Y = train_batch
#                     print("Train Batch:", train_X.shape, train_Y.shape)
#                     train_batch_count += 1
#                     print(f"Processed train batch {train_batch_count}/{total_batches}")

#                     # Example: model.train_step(train_X, train_Y)
#             if val_batch_count < total_batches:
#                 # Periodically validate
#                 val_batch = task.next_batch("val")
#                 if val_batch:
#                     val_X, val_Y = val_batch
#                     print("Val Batch:", val_X.shape, val_Y.shape)
#                     val_batch_count += 1
#                     print(f"Processed validation batch {val_batch_count}/{total_batches}")
                    
#                     # Example: model.validate_step(val_X, val_Y)

#     except StopIteration:
#         print("Completed all batches.")
#     finally:
#         print(f"Finished processing {train_batch_count} train batches and {val_batch_count} validation batches.")

