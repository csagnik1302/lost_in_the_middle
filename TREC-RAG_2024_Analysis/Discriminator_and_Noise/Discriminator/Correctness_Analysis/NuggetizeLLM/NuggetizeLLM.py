from transformers import AutoTokenizer, AutoModelForCausalLM
from nuggetizellm_prompt_creator import prompt_creator_nuggetizellm
import json
import gzip
import torch

#####################
with open(r'/home/irlab/sagnik/API_KEY','r') as f:
    hf_token=f.read()


model_name="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"


model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation="flash_attention_2")
tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

retr_set=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl', 'r') as f:
    for i in f:
        temp=json.loads(i)
        retr_set.append(temp)    

input_data=retr_set[3]

query_lookup_data=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/query_rag24.jsonl','r') as f:
    for i in f:
        query_lookup_data_temp=json.loads(i)
        query_lookup_data.append(query_lookup_data_temp)

prompt,query=prompt_creator_nuggetizellm(input_data,query_lookup_data)

# with open(r'C:\lost-in-the-middle\TREC-RAG_2024_Analysis\Discriminator_and_Noise\Discriminator\Data\Correctness_Analysis\misc\sample_prompt_nuggetizellm.json', 'r', encoding='utf-8') as f:
#     prompt = json.load(f)

input_processed=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)
input_size=inputs['input_ids'].shape[1]

output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, temperature=0.0, max_length=input_size+400, min_new_tokens=400)
output1=output_ids[0][input_size:]

output=tokenizer.decode(output1,skip_special_tokens=True)

print(output)

export_output={'query':query,'NuggetizeLLM_output':output}

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json','w') as f:
    json.dump(export_output,f,indent=2)


