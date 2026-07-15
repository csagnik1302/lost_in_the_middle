from transformers import AutoTokenizer, AutoModelForCausalLM
from gold_injector import gold_injector
from prompt_creator import prompt_creator
import json
import gzip
import torch
import re

#####################

def llm_output_generator(first_gold_position,corpus_lookup_index,model,tokenizer,retr_set_path):

    retr_set=[]

    with open(retr_set_path, 'r') as f:
        for i in f:
            temp=json.loads(i)
            retr_set.append(temp)    

    input_data=gold_injector(retr_set_path,first_gold_position,corpus_lookup_index)

    prompt,query=prompt_creator(input_data)

    query=query.replace('\n','')
    query=query.strip()


    input_processed=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)
    input_size=inputs['input_ids'].shape[1]


    output_ids=model.generate(inputs['input_ids'],tokenizer=tokenizer, do_sample=False, max_length=input_size+300, min_new_tokens=300, cache_implementation="offloaded")
    output1=output_ids[0][input_size:]

    output=tokenizer.decode(output1,skip_special_tokens=True)

    headings = [
        "references",
        "reference",
        "sources",
        "source",
        "citations",
        "citation",
        "bibliography",
        "works cited",
    ]


    pattern = (
        r"(?is)"
        r"\n+\s*(?:" + "|".join(re.escape(h) for h in headings) + r")\s*:?.*$"
    )

    output_text = re.sub(pattern, "", output).rstrip()
    output_text = output_text.replace("\n", " ")
    output_text = output_text.strip()


    return output_text, query


if __name__=='__main__':

    with open(r'/home/irlab/sagnik/API_KEY','r') as f:
        hf_token=f.read()

    retr_set_path=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl'

    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
    tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

    output_text, query=llm_output_generator(13,77,model,tokenizer,retr_set_path)

    print(output_text)

    export_output={'query':query,'generator_llm_output':output_text}

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_generator.json','w') as f:
        json.dump(export_output,f,indent=2)

