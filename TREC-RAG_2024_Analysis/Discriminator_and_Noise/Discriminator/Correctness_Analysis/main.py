from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import gzip
import torch
import re
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="bitsandbytes"
)

import sys
sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/Generator')
sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/NuggetizeLLM')
sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/NuggetizeScoreLLM')
sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/NuggetizeAssignerLLM')
sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/Evaluation')

from generator import llm_output_generator
from NuggetizeLLM import NuggetizeLLM
from NuggetizeScoreLLM import NuggetizeScoreLLM
from NuggetizeAssignerLLM import NuggetizeAssignerLLM
from evaluator import all_score, all_strict_score, vital_score, vital_strict_score, weighted_score, weighted_strict_score

#####################
with open(r'/home/irlab/sagnik/API_KEY','r') as f:
    hf_token=f.read()

model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
retr_set_path=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl'


model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

length_set=[]

with open(retr_set_path,'r') as f:
    for i in f:
        temp=json.loads(i)
        length_set.append(temp)  


context_length=len(length_set[0]['doc_discriminator'])+len(length_set[0]['doc_gold'])
corpus_length=len(length_set)


# ########################

test_indices=[0,14,28,43,57]
# test_indices=[0]

for i in tqdm(test_indices,desc='First Gold Doc Position'):         # Replace test_indices with range(context_length-2) for sliding window across computation
    for j in tqdm(range(corpus_length),desc='Corpus Index'):

        output_text, query=llm_output_generator(i,j,model,tokenizer,retr_set_path)
        output1=output_text

        nugget_dict=NuggetizeLLM(j,model,tokenizer,retr_set_path)
        output2=nugget_dict['NuggetizeLLM_output']

        score_list, nugget_list, query1=NuggetizeScoreLLM(model,tokenizer,nugget_dict)
        output3=score_list

        assigner_list, nugget_list1, query2=NuggetizeAssignerLLM(model,tokenizer,nugget_dict,output_text)
        output4=assigner_list


        ######## Eval

        all_score_out=all_score(assigner_list)
        all_strict_score_out=all_strict_score(assigner_list)

        vital_score_out=vital_score(assigner_list,score_list)
        vital_strict_score_out=vital_strict_score(assigner_list,score_list)

        weighted_score_out=weighted_score(assigner_list,score_list)
        weighted_strict_score_out=weighted_strict_score(assigner_list,score_list)

        output5={'all_score':all_score_out,'all_strict_score':all_strict_score_out,'vital_score':vital_score_out,'vital_strict_score':vital_strict_score_out,'weighted_score':weighted_score_out,'weighted_strict_score':weighted_strict_score_out}

        final_output={'first_gold_doc_pos':i,'corpus_position':j,'query':query,'generator_output':output1,'nuggetizellm_output':output2,'nuggetizescorellm_output':output3,'nuggetizeassignerllm_output':output4,'scores':output5}

        with open(r"/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_output.jsonl", "a") as f:
            f.write(json.dumps(final_output) + "\n")
            f.flush()
            os.fsync(f.fileno())





