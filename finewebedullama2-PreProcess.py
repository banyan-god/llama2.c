from datasets import load_dataset
import torch
import transformers 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
import sys



class Task:
    def __init__(self,batch_size,device,block_size):
        self.device=device
        self.block_size = block_size
        self.batch_size=batch_size

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = self.tokenizer.vocab_size


        self.initialized = False
        self.initialize()


    def initialize(self):

        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=False,num_proc=20)
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.1).values()

        
        columns_to_remove = ['id', 'url', 'text', 'dump', 'file_path', 'language', 'language_score', 'token_count', 'score', 'int_score']
        print("Tokenizing train");
        tokenized_datasets = train_dataset.map(self.tokenize_function, batched=True, num_proc=20, remove_columns=columns_to_remove)
        tokenized_datasets.save_to_disk('/workspace/data/tokenized_datasets')
        print("Tokenizing val");
        tokenized_val_datasets = val_dataset.map(self.tokenize_function, batched=True, num_proc=20,remove_columns=columns_to_remove)
        tokenized_val_datasets.save_to_disk('/workspace/data/tokenized_val_datasets')



    def tokenize_function(self, examples):
        # Tokenize the text (without adding special tokens)
        tokenized_output = self.tokenizer(examples['text'], truncation=True, max_length=self.block_size-1, add_special_tokens=False)
        
        # Manually add BOS and EOS tokens to each example
        input_ids = []
        for ids in tokenized_output['input_ids']:
            # Add bos_token_id at the start and eos_token_id at the end
            input_ids.append([self.tokenizer.bos_token_id] + ids)
        
        # Return the updated input_ids
        return {'input_ids': input_ids}


        
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task=Task(8,device,1024);


