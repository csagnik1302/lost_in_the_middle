# import json

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','r') as f:
#     out=json.load(f)

# for i in range(3):
#     for j in range(2):
#         out['i']+=1
#         out['j']+=1

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','w') as f:
#     json.dump(out,f)

# print(out)

# a="hi\"i"

# print(a.count('"'))


# import json

# input=[]

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_error_log_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         input.append(json.loads(i))

# indices_missed=[]

# for i in input:
#     indices_missed.append([i['first_gold_doc_pos'],i['corpus_position']])


# for i,j in indices_missed:
#     print(f'{i},{j}')

# ##########################################################################################################################################
# ##########################################################################################################################################


# import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import json
# import gzip
# import torch
# import re
# import os
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings(
#     "ignore",
#     category=FutureWarning,
#     module="bitsandbytes"
# )
# import sys

# input=[]

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_error_log_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         input.append(json.loads(i))

# indices_missed=[]

# for i in input:
#     indices_missed.append([i['first_gold_doc_pos'],i['corpus_position']])

# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/Generator')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/NuggetizeLLM')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/NuggetizeScoreLLM')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/NuggetizeAssignerLLM')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/Evaluation')

# from generator import llm_output_generator
# from NuggetizeLLM import NuggetizeLLM
# from NuggetizeScoreLLM import NuggetizeScoreLLM
# from NuggetizeAssignerLLM import NuggetizeAssignerLLM
# from evaluator import all_score, all_strict_score, vital_score, vital_strict_score, weighted_score, weighted_strict_score

# #####################
# with open(r'/home/irlab/sagnik/API_KEY','r') as f:
#     hf_token=f.read()

# retr_set_path=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl'

# model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
# model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
# tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

# length_set=[]

# with open(retr_set_path,'r') as f:
#     for i in f:
#         temp=json.loads(i)
#         length_set.append(temp)  


# # ########################

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/itr_index.json','r') as f:
#     itr=json.load(f)

# # test_indices=[0,6,13,19,25,32,38,44,51,57]
# # # test_indices=[0]

# for i,j in indices_missed:

#     # if i<itr['i']:
#     #     continue

#     # if i==itr['i'] and j<=itr['j']:
#     #     continue
#     output_text, query=llm_output_generator(i,j,model,tokenizer,retr_set_path)
#     output1=output_text
#     print('DONE')

#     nugget_dict=NuggetizeLLM(j,model,tokenizer,retr_set_path)
#     output2=nugget_dict['NuggetizeLLM_output']
#     print('DONE')

#     score_list, nugget_list, query1=NuggetizeScoreLLM(model,tokenizer,nugget_dict)
#     output3=score_list
#     print('DONE')

#     assigner_list, nugget_list1, query2=NuggetizeAssignerLLM(model,tokenizer,nugget_dict,output_text)
#     output4=assigner_list
#     print('DONE')

#     ######## Eval

#     all_score_out=all_score(assigner_list)
#     all_strict_score_out=all_strict_score(assigner_list)

#     vital_score_out=vital_score(assigner_list,score_list)
#     vital_strict_score_out=vital_strict_score(assigner_list,score_list)

#     weighted_score_out=weighted_score(assigner_list,score_list)
#     weighted_strict_score_out=weighted_strict_score(assigner_list,score_list)

#     output5={'all_score':all_score_out,'all_strict_score':all_strict_score_out,'vital_score':vital_score_out,'vital_strict_score':vital_strict_score_out,'weighted_score':weighted_score_out,'weighted_strict_score':weighted_strict_score_out}

#     final_output={'model':model_name,'first_gold_doc_pos':i,'corpus_position':j,'query':query,'generator_output':output1,'nuggetizellm_output':output2,'nuggetizescorellm_output':output3,'nuggetizeassignerllm_output':output4,'scores':output5}


#     with open(fr"/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_{model_name[model_name.index('/')+1:]}.jsonl", "a") as f:
#         f.write(json.dumps(final_output) + "\n")
#         f.flush()
#         os.fsync(f.fileno())

# ##########################################################################################################################################
# ##########################################################################################################################################

import json
import re

out=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
    for i in f:
        out.append(json.loads(i))

# out={"model": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "first_gold_doc_pos": 13, "corpus_position": 77, "query": "why did desoto fail", "generator_output": "The DeSoto brand failed for several reasons. One of the main reasons was the 1958 recession, which affected demand for mid-priced automobiles. DeSoto sales were particularly affected, and they failed to recover in 1959 and 1960. Another factor was the competition from other Chrysler brands, such as Plymouth, which offered similar models at a lower price. Additionally, the DeSoto brand had a weak dealer network, as many dealers chose to sell higher-volume Plymouth models instead of slower-selling DeSoto models. Finally, the brand management at Chrysler may have contributed to the failure of DeSoto, as they pitted each of the five Chrysler brands against one another, rather than carefully managing the market to specific price points for all consumers. This lack of focus may have led to confusion among consumers and a lack of clear brand identity for DeSoto.", "nuggetizellm_output": ["1958 recession hurt DeSoto sales\", \"DeSoto sales failed to recover in 1959 and 1960\", \"DeSoto models similar to concurrent Chryslers\", \"Rumors of DeSoto discontinuation\", \"Weakened DeSoto dealer network\", \"Chrysler spun Plymouth off into standalone dealerships\", \"Dealers chose higher-volume Plymouth over DeSoto\", \"DeSoto failed to adjust to changing market trends\", \"No new compact car model in 1960\", \"Chrysler's brand management pitted divisions against each other\", \"Lack of careful market management\", \"General Motors had successful market planning\", \"DeSoto's failure hastened its demise\", \"Compounded by 1961 Newport model introduction\", \"Newport model was an upper-tier DeSoto competitor\", \"DeSoto brand pushed to the brink in 1961"], "nuggetizescorellm_output": ["vital"], "nuggetizeassignerllm_output": ["support"], "scores": {"all_score": 1.0, "all_strict_score": 1.0, "vital_score": 1.0, "vital_strict_score": 1.0, "weighted_score": 1.0, "weighted_strict_score": 1.0}}

# import json
# import re

# INPUT_PATH = r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json'
# OUTPUT_PATH = r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm_fixed.json'


def split_glued_nuggets(raw):
    """
    Handles the case where NuggetizeLLM_output is a list containing ONE
    string with all nuggets joined by '", "' (the bug from the buggy
    parser). Splits it back into individual nugget strings.
    """
    # Strip one layer of leading/trailing quote if present
    raw = raw.strip()
    if raw.startswith('"'):
        raw = raw[1:]
    if raw.endswith('"'):
        raw = raw[:-1]

    # Split on the boundary between nuggets: a quote, comma, space, quote
    parts = re.split(r'"\s*,\s*"', raw)

    # Clean whitespace/newlines inside each part
    parts = [p.replace('\n', ' ').strip() for p in parts]
    parts = [p for p in parts if p]

    return parts


# Possible key names this field has shown up under across different files
NUGGET_KEY_CANDIDATES = [
    'NuggetizeLLM_output',
    'nuggetizellm_output',
]


def find_nugget_key(nugget_dict):
    for key in NUGGET_KEY_CANDIDATES:
        if key in nugget_dict:
            return key
    raise KeyError(
        f"None of the expected keys {NUGGET_KEY_CANDIDATES} found in dict. "
        f"Available keys: {list(nugget_dict.keys())}"
    )


def get_nugget_list(nugget_dict):
    key = find_nugget_key(nugget_dict)
    raw_output = nugget_dict[key]

    # Already a proper list of separate nuggets -> nothing to fix
    if isinstance(raw_output, list) and len(raw_output) > 1:
        return raw_output, key

    # List with a single glued-together string -> needs splitting
    if isinstance(raw_output, list) and len(raw_output) == 1:
        return split_glued_nuggets(raw_output[0]), key

    # Raw string (not even wrapped in a list) -> needs splitting
    if isinstance(raw_output, str):
        return split_glued_nuggets(raw_output), key

    raise ValueError(f"Unexpected type for {key}: {type(raw_output)}")

import os

if __name__ == '__main__':
    # with open(INPUT_PATH, 'r') as f:
    #     nugget_dict = json.load(f)

    for i in out:

        nugget_list, key = get_nugget_list(i)
    
        if len(nugget_list)!=len(i['nuggetizescorellm_output']):
            gen={"first_gold_doc_pos": i['first_gold_doc_pos'], "corpus_position": i['corpus_position']}
            

            with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_fixing_need_log.jsonl','a') as f:
                f.write(json.dumps(gen) + "\n")
                f.flush()
                os.fsync(f.fileno())



############

# import json
# import re

# out=[]

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         out.append(json.loads(i))

# # out={"model": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "first_gold_doc_pos": 13, "corpus_position": 77, "query": "why did desoto fail", "generator_output": "The DeSoto brand failed for several reasons. One of the main reasons was the 1958 recession, which affected demand for mid-priced automobiles. DeSoto sales were particularly affected, and they failed to recover in 1959 and 1960. Another factor was the competition from other Chrysler brands, such as Plymouth, which offered similar models at a lower price. Additionally, the DeSoto brand had a weak dealer network, as many dealers chose to sell higher-volume Plymouth models instead of slower-selling DeSoto models. Finally, the brand management at Chrysler may have contributed to the failure of DeSoto, as they pitted each of the five Chrysler brands against one another, rather than carefully managing the market to specific price points for all consumers. This lack of focus may have led to confusion among consumers and a lack of clear brand identity for DeSoto.", "nuggetizellm_output": ["1958 recession hurt DeSoto sales\", \"DeSoto sales failed to recover in 1959 and 1960\", \"DeSoto models similar to concurrent Chryslers\", \"Rumors of DeSoto discontinuation\", \"Weakened DeSoto dealer network\", \"Chrysler spun Plymouth off into standalone dealerships\", \"Dealers chose higher-volume Plymouth over DeSoto\", \"DeSoto failed to adjust to changing market trends\", \"No new compact car model in 1960\", \"Chrysler's brand management pitted divisions against each other\", \"Lack of careful market management\", \"General Motors had successful market planning\", \"DeSoto's failure hastened its demise\", \"Compounded by 1961 Newport model introduction\", \"Newport model was an upper-tier DeSoto competitor\", \"DeSoto brand pushed to the brink in 1961"], "nuggetizescorellm_output": ["vital"], "nuggetizeassignerllm_output": ["support"], "scores": {"all_score": 1.0, "all_strict_score": 1.0, "vital_score": 1.0, "vital_strict_score": 1.0, "weighted_score": 1.0, "weighted_strict_score": 1.0}}

# # import json
# # import re

# # INPUT_PATH = r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json'
# # OUTPUT_PATH = r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm_fixed.json'


# def split_glued_nuggets(raw):
#     """
#     Handles the case where NuggetizeLLM_output is a list containing ONE
#     string with all nuggets joined by '", "' (the bug from the buggy
#     parser). Splits it back into individual nugget strings.
#     """
#     # Strip one layer of leading/trailing quote if present
#     raw = raw.strip()
#     if raw.startswith('"'):
#         raw = raw[1:]
#     if raw.endswith('"'):
#         raw = raw[:-1]

#     # Split on the boundary between nuggets: a quote, comma, space, quote
#     parts = re.split(r'"\s*,\s*"', raw)

#     # Clean whitespace/newlines inside each part
#     parts = [p.replace('\n', ' ').strip() for p in parts]
#     parts = [p for p in parts if p]

#     return parts


# # Possible key names this field has shown up under across different files
# NUGGET_KEY_CANDIDATES = [
#     'NuggetizeLLM_output',
#     'nuggetizellm_output',
# ]


# def find_nugget_key(nugget_dict):
#     for key in NUGGET_KEY_CANDIDATES:
#         if key in nugget_dict:
#             return key
#     raise KeyError(
#         f"None of the expected keys {NUGGET_KEY_CANDIDATES} found in dict. "
#         f"Available keys: {list(nugget_dict.keys())}"
#     )


# def get_nugget_list(nugget_dict):
#     key = find_nugget_key(nugget_dict)
#     raw_output = nugget_dict[key]

#     # Already a proper list of separate nuggets -> nothing to fix
#     if isinstance(raw_output, list) and len(raw_output) > 1:
#         return raw_output, key

#     # List with a single glued-together string -> needs splitting
#     if isinstance(raw_output, list) and len(raw_output) == 1:
#         return split_glued_nuggets(raw_output[0]), key

#     # Raw string (not even wrapped in a list) -> needs splitting
#     if isinstance(raw_output, str):
#         return split_glued_nuggets(raw_output), key

#     raise ValueError(f"Unexpected type for {key}: {type(raw_output)}")

# import os

# if __name__ == '__main__':
#     # with open(INPUT_PATH, 'r') as f:
#     #     nugget_dict = json.load(f)
    
#     out=[]

#     with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#         for i in f:
#             out.append(json.loads(i))
    
#     out1=out.copy()

#     for i in out:

#         nugget_list, key = get_nugget_list(i)
    
#         if len(nugget_list)!=len(i['nuggetizescorellm_output']):

#             out1.remove(i)

        
#         with open(fr"/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/fix.jsonl", "w") as f:
            
#             for i in out1:
            
#                 f.write(json.dumps(i) + "\n")
            

