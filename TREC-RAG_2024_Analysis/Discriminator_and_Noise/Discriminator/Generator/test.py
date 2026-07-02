# import json

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Misc/sample_prompt.json','r') as f:
#     out=json.load(f)

# print(out)

from transformers import AutoTokenizer, AutoModelForCausalLM
from gold_injector import gold_injector
from prompt_creator import prompt_creator
import json
import gzip
import torch

#####################
with open(r'/home/irlab/sagnik/API_KEY','r') as f:
    hf_token=f.read()


model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"


model=AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="sdpa",device_map={"": 0})
tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

retr_set=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl', 'r') as f:
    for i in f:
        temp=json.loads(i)
        retr_set.append(temp)    

input_data=gold_injector(retr_set,1)[0]

query_lookup_data=[]

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/query_rag24.jsonl','r') as f:
#     for i in f:
#         query_lookup_data_temp=json.loads(i)
#         query_lookup_data.append(query_lookup_data_temp)

# prompt=prompt_creator(input_data,query_lookup_data)

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Misc/sample_prompt.json', 'r', encoding='utf-8') as f:
    prompt = json.load(f)

input_processed=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)

token_count = inputs["input_ids"].shape[1]

print(token_count)