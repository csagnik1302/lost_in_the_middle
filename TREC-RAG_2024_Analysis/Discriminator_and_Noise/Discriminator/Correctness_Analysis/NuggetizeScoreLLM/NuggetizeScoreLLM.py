NuggetizeScoreLLM.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from nuggetizescorellm_prompt_creator import prompt_creator_nuggetizescorellm
import json
import gzip
import torch
import ast
import math

#####################
with open(r'/home/irlab/sagnik/API_KEY','r') as f:
    hf_token=f.read()


model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"


model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation="flash_attention_2")
tokenizer=AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True, token=hf_token)

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json','r') as f:
    nugget_dict=json.load(f)

nugget_list_temp=nugget_dict['NuggetizeLLM_output']

for i in range(len(nugget_list_temp)):
    if nugget_list_temp[i]=='[':
        starting_ind=i
    if nugget_list_temp[i]==']':
        ending_ind=i

nugget_list=ast.literal_eval(nugget_list_temp[starting_ind:ending_ind+1])

prompt_list=[]

for i in nugget_list:
    prompt,query=prompt_creator_nuggetizescorellm(nugget_dict,i)

    prompt_list.append(prompt)


final_output_list=[]

for i in prompt_list:
    input_processed=tokenizer.apply_chat_template(i,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)
    input_size=inputs['input_ids'].shape[1]

    output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, temperature=0.0, max_length=input_size+10, min_new_tokens=10)
    output1=output_ids[0][input_size:]

    output=tokenizer.decode(output1,skip_special_tokens=True)

    for i in output.split():
        j=i.lower()
        if j=='vital':
            temp='vital'
            final_output_list.append(temp)
            break
        if j=='okay':
            temp='okay'
            final_output_list.append(temp)
            break

    
export_output={'query':query,'nugget_list':nugget_list,'NuggetizeLLM_output':final_output_list}

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizescorellm.json','w') as f:
    json.dump(export_output,f,indent=2)

print(final_output_list)


