from datasets import load_dataset
import torch
import transformers 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk



class Task:
    def __init__(self,batch_size,device,block_size):
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
        
        
    def initialize(self):

        train_dataset = load_from_disk('/tmp/tokenized_datasets')
        val_dataset = load_from_disk('/tmp/tokenized_val_datasets')
        train_dataset = train_dataset.with_format("torch")
        val_dataset = val_dataset.with_format("torch")


        # tokenized_datasets = tokenized_datasets.with_format("torch")
        # tokenized_val_datasets=tokenized_val_datasets.with_format("torch")
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size,sampler=val_sampler,  pin_memory=True)
        self.initialized = True
        
    def tokenize_function(self,tokenized_output):

        input_ids = tokenized_output['input_ids'].long()
        
        # Create labels by shifting input_ids and appending -100 at the end of each sequence
        labels = torch.cat([input_ids[:, 1:], torch.full((input_ids.size(0), 1), -100, dtype=torch.long)], dim=1)
        
        # Package the processed data back into the dictionary
        tokenized_output['input_ids'] = input_ids
        tokenized_output['labels'] = labels
        
        return tokenized_output
    def iter_batches(self, split, start_index=0):
        if self.initialized == False:
            self.initialize()
        if split == "train":
            data_loader = self.train_loader
        elif split == "val":
            data_loader = self.val_loader
        else:
            raise ValueError("Invalid split name. Use 'train' or 'val'.")
        batch_iterator = iter(data_loader)
        # Skip batches up to the specified start index
        for _ in range(start_index):
            next(batch_iterator, None) 
            
        for batch in data_loader:
            try:
                yield self.process_batch(batch)
            except Exception as e:
                print("Error processing batch:", e)
                break 
    def process_batch(self, batch):
        batch=self.tokenize_function(batch)
        X = batch['input_ids'].detach().to(self.device, non_blocking=True)
        Y = batch['labels'].detach().to(self.device, non_blocking=True)
        return X, Y
        
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task=Task(8,device,1024);
    split='train'
    batch_generator = task.iter_batches(split)
    X, Y = next(batch_generator)  # Get the first batch
    print(X.shape, Y.shape) 

