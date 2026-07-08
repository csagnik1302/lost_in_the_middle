from transformers import AutoTokenizer, AutoModelForCausalLM
from nuggetizeassignerllm_prompt_creator import prompt_creator_nuggetizeassignerllm
import json
import gzip
import torch
import ast
import math

#####################

def NuggetizeAssignerLLM(model,tokenizer,nugget_dict,generator_output_passage):

    nugget_list=nugget_dict['NuggetizeLLM_output']

    prompt_list=[]

    for i in nugget_list:
        prompt,query=prompt_creator_nuggetizeassignerllm(nugget_dict,generator_output_passage,i)

        prompt_list.append(prompt)


    final_output_list=[]

    for i in prompt_list:
        input_processed=tokenizer.apply_chat_template(i,tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)
        input_size=inputs['input_ids'].shape[1]

        output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, max_length=input_size+10, min_new_tokens=10)
        output1=output_ids[0][input_size:]

        output=tokenizer.decode(output1,skip_special_tokens=True)

        for i in output.split():
            j=i.lower()
            if j=='support':
                temp='support'
                final_output_list.append(temp)
                break
            if j=='partial_support':
                temp='partial_support'
                final_output_list.append(temp)
                break
            if j=='not_support':
                temp='not_support'
                final_output_list.append(temp)
                break
        
    
    return final_output_list, nugget_list, query



if __name__=='__main__':

    with open(r'C:\lost-in-the-middle\API_KEY','r') as f:
        hf_token=f.read()


    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"


    model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation="flash_attention_2")
    tokenizer=AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True, token=hf_token)

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json','r') as f:
        nugget_dict=json.load(f)

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_generator.json','r') as f:
        out=json.load()
    
    passage=out['generator_llm_output']
    

    final_output_list,nugget_list,query=NuggetizeAssignerLLM(model,tokenizer,nugget_dict,passage)


    export_output={'query':query,'nugget_list':nugget_list,'NuggetizeAssigner_output':final_output_list}

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc','w') as f:
        json.dump(export_output,f,indent=2)

    print(final_output_list)

