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


model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation="sdpa")
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
input_size=inputs['input_ids'].shape[1]

output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, temperature=0.0, max_length=input_size+400, min_new_tokens=400)
output1=output_ids[0][input_size:]

output=tokenizer.decode(output1,skip_special_tokens=True)

print(output)

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Misc/sample_output.txt','w') as f:
    f.write(output)


