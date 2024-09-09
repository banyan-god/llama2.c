from datasets import load_dataset
import torch
import transformers 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from datasets import load_from_disk

def custom_collate_fn(batch, block_size, tokenize_fn, pad_token_id):
    # Flatten the batch list to create a long list of sequences
    batch = [item for sublist in batch for item in sublist['input_ids']]
    
    input_ids = []
    labels = []
    while len(batch) >= block_size + 1:  # +1 to account for both input and label
        current_input_ids = torch.tensor(batch[:block_size + 1])
        input_ids.append(current_input_ids[:-1])
        labels.append(current_input_ids[1:])
        
        # Remove those tokens from the batch list
        batch = batch[block_size:]
    
    # Drop any remaining tokens if they don't form a full block
    if len(batch) < block_size + 1:
        batch = []  # This effectively drops the last incomplete sequence

    # Stack the inputs and labels
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    
    # Create the dictionary format expected by tokenize_function
    tokenized_output = {'input_ids': input_ids, 'labels': labels}
    
    # Pass the batch through the tokenize_function
    tokenized_output = tokenize_fn(tokenized_output)
    
    return tokenized_output




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
        #dataset from finewebedullama2-preprocess 
        train_dataset = load_from_disk('/workspace/data/tokenized_datasets')
        val_dataset = load_from_disk('/workspace/data/tokenized_val_datasets')
        train_dataset = train_dataset.with_format("torch")
        val_dataset = val_dataset.with_format("torch")


        # tokenized_datasets = tokenized_datasets.with_format("torch")
        # tokenized_val_datasets=tokenized_val_datasets.with_format("torch")
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=8,prefetch_factor=8, persistent_workers=True, pin_memory=True ,     collate_fn=lambda batch: custom_collate_fn(batch, self.block_size, self.tokenize_function, self.tokenizer.pad_token_id))
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size,sampler=val_sampler, num_workers=8,prefetch_factor=8, persistent_workers=True,  pin_memory=True, 
      collate_fn=lambda batch: custom_collate_fn(batch, self.block_size, self.tokenize_function, self.tokenizer.pad_token_id))
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
        # batch=self.tokenize_function(batch)
        X = batch['input_ids'].detach().to(self.device, non_blocking=True)
        Y = batch['labels'].detach().to(self.device, non_blocking=True)
        return X, Y



def test_batch_block_size(print_every=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = Task(batch_size=8, device=device, block_size=1024)
    split = 'train'
    batch_generator = task.iter_batches(split)
    
    for i, (X, Y) in enumerate(batch_generator):
        assert X.shape[1] == 1024, f"X block size is {X.shape[1]} in batch {i}, expected 1024"
        assert Y.shape[1] == 1024, f"Y block size is {Y.shape[1]} in batch {i}, expected 1024"
        if i % print_every == 0:
            print(f"\rBatch {i} passed: X and Y block sizes are 1024.", end="")
    
    # Final newline after loop completes
    print("\nAll batches processed successfully.")




        
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task=Task(8,device,1024);
    split='train'
    batch_generator = task.iter_batches(split)
    X, Y = next(batch_generator)  # Get the first batch
    print(X.shape, Y.shape) 
    test_batch_block_size()

