from datasets import load_dataset
import torch
import transformers 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist



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
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", split="train", streaming=False,num_proc=16)
        split_dataset = dataset.train_test_split(test_size=0.1)  # Split 10% for testing, 90% for training

        # Access the training and validation datasets
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        
        tokenized_datasets = train_dataset.map(self.tokenize_function, batched=True, remove_columns={'id','url','text','dump','date','file_path','language','language_score','token_count'})
        tokenized_val_datasets = val_dataset.map(self.tokenize_function, batched=True,remove_columns={'id','url','text','dump','date','file_path','language','language_score','token_count'})

                # Create samplers for distributed training
        train_sampler = DistributedSampler(tokenized_datasets, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(tokenized_val_datasets, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        
        # tokenized_datasets = tokenized_datasets.with_format("torch")
        # tokenized_val_datasets=tokenized_val_datasets.with_format("torch")
        self.train_loader = DataLoader(tokenized_datasets, batch_size=self.batch_size,  sampler=train_sampler, pin_memory=True)
        self.val_loader = DataLoader(tokenized_val_datasets, batch_size=self.batch_size, sampler=val_sampler, pin_memory=True)
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
                continue 
    def process_batch(self, batch):
        X = batch['input_ids'].clone().detach().to(self.device, non_blocking=True)
        Y = batch['labels'].clone().detach().to(self.device, non_blocking=True)
        return X, Y
        
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task=Task(10,device,1024);
    split='val'
    batch_generator = task.iter_batches(split)
    X, Y = next(batch_generator)  # Get the first batch
    print(X.shape, Y.shape) 

