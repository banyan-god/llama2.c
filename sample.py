"""
Sample from the trained model with PyTorch
"""
import os
import safetensors.torch as st
from safetensors import safe_open
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
import json

import transformers 
from tinystories import get_tokenizer_model_path
import random

# -----------------------------------------------------------------------------
checkpoint = 'out/model_state_dict.safetensors'
start = "Life and game of thrones" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 600 # number of tokens generated in each sample
temperature = 1.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed =0
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
#dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
print(device)

# init from a model saved in a specific directory
state_dict={}
with safe_open(checkpoint, framework="pt",  device=device) as f:
    for key in f.keys():
       state_dict[key] = f.get_tensor(key)
    metadata = f.metadata()


print("Loaded metadata:", metadata.keys())
print(metadata['config'])

model_args = json.loads(metadata['model_args'])
config=json.loads(metadata['config'])


gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_source = config.get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = transformers.AutoTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = enc.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('---------------')
