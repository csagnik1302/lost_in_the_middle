from transformers import AutoTokenizer, AutoModelForCausalLM
from gold_injector import gold_injector
from prompt_creator import prompt_creator
import json
import gzip
import torch

#####################
with open(r'/home/irlab/sagnik/API_KEY','r') as f:
    hf_token=f.read()


model_name="mistralai/Mistral-7B-Instruct-v0.3"


model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,torch_dtype=torch.bfloat16,attn_implementation="sdpa")
tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

# retr_set=[]

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl', 'r') as f:
#     for i in f:
#         temp=json.loads(i)
#         retr_set.append(temp)    

# input_data=gold_injector(retr_set,1)[0]

# query_lookup_data=[]

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/query_rag24.jsonl','r') as f:
#     for i in f:
#         query_lookup_data_temp=json.loads(i)
#         query_lookup_data.append(query_lookup_data_temp)

# prompt=prompt_creator(input_data,query_lookup_data)

prompt=''
with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Misc/sample_prompt.txt','r') as f:
    out=f.readlines()
    for i in out:
        prompt+=i

inputs=tokenizer(prompt,return_tensors="pt")
input_size=inputs['input_ids'].shape[1]

output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, temperature=0.0, min_length=400, min_new_tokens=100)
output1=output_ids[0][input_size:]

output=tokenizer.decode(output1,skip_special_tokens=True)

print(output)

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Misc/sample_output.txt','w') as f:
    f.write(output)


