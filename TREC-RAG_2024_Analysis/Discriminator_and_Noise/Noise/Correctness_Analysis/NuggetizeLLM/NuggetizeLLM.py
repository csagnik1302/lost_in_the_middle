from transformers import AutoTokenizer, AutoModelForCausalLM
from nuggetizellm_prompt_creator import prompt_creator_nuggetizellm_noise
import json
import gzip
import torch
import ast

#####################

def NuggetizeLLM(corpus_lookup_index,model,tokenizer,retr_set_path):

    retr_set=[]

    with open(retr_set_path, 'r') as f:
        for i in f:
            temp=json.loads(i)
            retr_set.append(temp)    

    input_data=retr_set[corpus_lookup_index]

    prompt,query=prompt_creator_nuggetizellm_noise(input_data)

    query=query.replace('\n','')

    input_processed=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)
    input_size=inputs['input_ids'].shape[1]

    output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, max_length=input_size+1100, min_new_tokens=1100, cache_implementation="offloaded")
    output1=output_ids[0][input_size:]

    output=tokenizer.decode(output1,skip_special_tokens=True)

    starting_Flag=False
    ending_Flag=False

    for i in range(len(output)):
        if output[i]=='[':
            starting_ind=i
            starting_Flag=True
        if output[i]==']':
            ending_ind=i
            ending_Flag=True
        
        if starting_Flag==True and ending_Flag==True:
            break

    raw_list=output[starting_ind:ending_ind+1].strip().strip('[]').strip()

    nugget_list=[]

    for i in raw_list.splitlines():
        j=i.strip()

        if not j:
            continue

        if j.endswith(","):
            j=j[:-1].strip()

        if not j:
            continue        
        
        if j.startswith('"') and j.endswith('"'):
            j=j[1:-1]

        nugget_list.append(j)

    nugget_dict={'query':query,'NuggetizeLLM_output':nugget_list}
    
    return nugget_dict

    

if __name__=="__main__":


    with open(r'/home/irlab/sagnik/API_KEY','r') as f:
        hf_token=f.read()

    retr_set_path=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl'

    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
    tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

    nugget_dict=NuggetizeLLM(4,model,tokenizer,retr_set_path)

    print(nugget_dict)

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json','w') as f:
        json.dump(nugget_dict,f,indent=2)

