import threading
from collections import deque
from datasets import load_dataset
import torch
import transformers 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from datasets.distributed import split_dataset_by_node


class Task:
    def __init__(self,batch_size,device,block_size, prefetch_buffer=5):
        self.device=device
        self.block_size = block_size
        self.batch_size=batch_size

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
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", split="train", streaming=False,num_proc=16)
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
                    # Convert to CPU tensors to save GPU memory
                    cpu_batch = {k: v.cpu() for k, v in batch.items()}
                    queue.append(cpu_batch)


                
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
            
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = Task(10, device, 1024)
    for _ in range(5):  # Get some batches
        X, Y = task.next_batch("train")
        if X is not None and Y is not None:
            print("Train Batch:", X.shape, Y.shape)
        X, Y = task.next_batch("val")
        if X is not None and Y is not None:
            print("Val Batch:", X.shape, Y.shape)

            # Fetch a batch for training                print("Train Batch:", 
