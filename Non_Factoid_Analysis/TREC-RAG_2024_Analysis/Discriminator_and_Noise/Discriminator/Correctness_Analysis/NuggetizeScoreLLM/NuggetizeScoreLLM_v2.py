from sentence-transformers import CrossEncoder
from nuggetizescorellm_prompt_creator import prompt_creator_nuggetizescorellm
import json
import gzip
import torch
import ast
import math
import re
import torch

#####################


def NuggetizeScoreLLM(model, tokenizer, nugget_dict, query):

    model=CrossEncoder(model_name, activation_fn=torch,nn.Sigmoid())

    nugget_list = nugget_dict['NuggetizeLLM_output']
    query = nugget_dict['query']
    
    model_input_list=[(query,i) for i in nugget_list]
    
    scores=model.predict(model_input_list)

    scores=[]

    return final_output_list, nugget_list, query


if __name__=='__main__':

    with open(r'/home/irlab/sagnik/API_KEY','r') as f:
        hf_token=f.read()


    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"


    model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation="flash_attention_2")
    tokenizer=AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True, token=hf_token)


    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json','r') as f:
        nugget_dict=json.load(f)
    
    score_output_list=NuggetizeScoreLLM(model,tokenizer,nugget_dict)
    
    score_output_list, nugget_list, query=NuggetizeScoreLLM(model,tokenizer,nugget_dict)
    
    export_output={'query':query,'nugget_list':nugget_list,'NuggetizeLLM_output':score_output_list}

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizescorellm.json','w') as f:
        json.dump(export_output,f,indent=2)

    print(score_output_list)