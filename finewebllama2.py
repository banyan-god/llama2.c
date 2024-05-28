from datasets import load_dataset
import torch
import transformers 
from torch.utils.data import DataLoader



class Task:
    def __init__(self,batch_size,device,block_size):
        self.device=device
        self.block_size = block_size
        self.batch_size=batch_size

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = self.tokenizer.vocab_size
        self.initialized = False
        
        
    def initialize(self):
        dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split='train', streaming=False,num_proc=28)
        iterable_dataset = dataset.to_iterable_dataset()
        shuffled_dataset = iterable_dataset.shuffle(buffer_size=100000)
          # Skip the first 1000 records for validation
        val_dataset = shuffled_dataset.take(10000) 
        train_dataset = shuffled_dataset.skip(10000)
        
        tokenized_datasets = train_dataset.map(self.tokenize_function, batched=True, remove_columns={'id','url','text','dump','date','file_path','language','language_score','token_count'})
        tokenized_val_datasets = val_dataset.map(self.tokenize_function, batched=True,remove_columns={'id','url','text','dump','date','file_path','language','language_score','token_count'})
        
        # tokenized_datasets = tokenized_datasets.with_format("torch")
        # tokenized_val_datasets=tokenized_val_datasets.with_format("torch")
        self.train_loader = DataLoader(tokenized_datasets, batch_size=self.batch_size,  pin_memory=True)
        self.val_loader = DataLoader(tokenized_val_datasets, batch_size=self.batch_size, pin_memory=True)
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

