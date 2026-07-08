from transformers import AutoTokenizer, AutoModelForCausalLM
from nuggetizellm_prompt_creator import prompt_creator_nuggetizellm
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

    prompt,query=prompt_creator_nuggetizellm(input_data)

    query=query.replace('\n','')

    input_processed=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)
    input_size=inputs['input_ids'].shape[1]

    output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, max_length=input_size+400, min_new_tokens=400)
    output1=output_ids[0][input_size:]

    output=tokenizer.decode(output1,skip_special_tokens=True)

    for i in range(len(output)):
        if output[i]=='[':
            starting_ind=i
        if output[i]==']':
            ending_ind=i

    nugget_list=ast.literal_eval(output[starting_ind:ending_ind+1])

    nugget_dict={'query':query,'NuggetizeLLM_output':nugget_list}
    
    return nugget_dict


if __name__=="__main__":


    with open(r'C:\lost-in-the-middle\API_KEY','r') as f:
        hf_token=f.read()

    retr_set_path=r'C:\lost-in-the-middle\TREC-RAG_2024_Analysis\Discriminator_and_Noise\Discriminator\Data\generator_input_data_gold_fixed_3.jsonl'

    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
    tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

    nugget_dict=NuggetizeLLM(0,model,tokenizer,retr_set_path)

    print(nugget_dict)

    with open(r'C:\lost-in-the-middle\TREC-RAG_2024_Analysis\Discriminator_and_Noise\Discriminator\Correctness_Analysis\misc\sample_output_nuggetizellm.json','w') as f:
        json.dump(nugget_dict,f,indent=2)

