##########################
# RETRY SCRIPT: re-runs the cases that failed in the main pipeline script
# SET APPROPRIATE METHOD BEFORE STARTING
##########################

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import gzip
import torch
import re
import os
from tqdm import tqdm
from setproctitle import setproctitle
from dotenv import load_dotenv

setproctitle("Discriminator-corr")

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="bitsandbytes"
)

import sys
sys.path.insert(0,r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/Generator')
sys.path.insert(0,r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/NuggetizeLLM')
sys.path.insert(0,r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/NuggetizeScoreLLM')
sys.path.insert(0,r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/NuggetizeAssignerLLM')
sys.path.insert(0,r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/Evaluation')

from generator import llm_output_generator
from NuggetizeLLM import NuggetizeLLM
from NuggetizeScoreLLM import NuggetizeScoreLLM
from NuggetizeAssignerLLM import NuggetizeAssignerLLM
from evaluator import all_score, all_strict_score, vital_score, vital_strict_score, weighted_score, weighted_strict_score
import tomllib

#####################

load_dotenv()
hf_token=os.getenv('HF_TOKEN')

with open('config.toml', 'rb') as f:
    config=tomllib.load(f)

autonugg_model_name=config['model_setting']['model']
model_name=config['model_setting']['generator_model']
method='bm25'
gold_count=config['misc_setting']['gold_count']
total=config['misc_setting']['total']
count=config['misc_setting']['gold_pos_count']

BASE=r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise'
gen_short_name=model_name[model_name.index('/')+1:]

retr_set_path=rf'{BASE}/Data/{method}/generator_input_data_gold_fixed_{gold_count}_app2A.jsonl'

# Error log written by the main script (input of this retry script)
main_error_log_path=rf'{BASE}/Discriminator/Correctness_Analysis/misc/{method}/pipeline_error_log_gold_{gold_count}_{gen_short_name}_app2A.jsonl'
# Main script's output file (successful retries are appended here)
output_path=rf'{BASE}/Discriminator/Correctness_Analysis/misc/{method}/pipeline_output_gold_{gold_count}_{gen_short_name}_app2A.jsonl'
# Cases that fail again during the retry are logged here
retry_error_log_path=rf'{BASE}/Discriminator/Correctness_Analysis/misc/{method}/pipeline_error_log_gold_{gold_count}_{gen_short_name}_app2A_2nd.jsonl'

autonugg_model=AutoModelForCausalLM.from_pretrained(autonugg_model_name,token=hf_token,attn_implementation='flash_attention_2')
model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
autonugg_tokenizer=AutoTokenizer.from_pretrained(autonugg_model_name,token=hf_token)
tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)


# ########################

# Load the (first_gold_doc_pos, corpus_position) pairs that failed in the main script
# (deduplicated, original order preserved)
failed_cases=[]
seen=set()
with open(main_error_log_path,'r') as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        entry=json.loads(line)
        key=(entry['first_gold_doc_pos'],entry['corpus_position'])
        if key not in seen:
            seen.add(key)
            failed_cases.append(key)

print(f"Retrying {len(failed_cases)} failed cases")

for i,j in tqdm(failed_cases,desc='Retrying failed cases'):

    try:
        output_text, query=llm_output_generator(i,j,model,tokenizer,retr_set_path)
        output1=output_text

        nugget_dict=NuggetizeLLM(j,autonugg_model,autonugg_tokenizer,retr_set_path)
        output2=nugget_dict['NuggetizeLLM_output']

        score_list, nugget_list, query1=NuggetizeScoreLLM(autonugg_model,autonugg_tokenizer,nugget_dict)
        output3=score_list

        assigner_list, nugget_list1, query2=NuggetizeAssignerLLM(autonugg_model,autonugg_tokenizer,nugget_dict,output_text)
        output4=assigner_list


        ######## Eval

        all_score_out=all_score(assigner_list)
        all_strict_score_out=all_strict_score(assigner_list)

        vital_score_out=vital_score(assigner_list,score_list)
        vital_strict_score_out=vital_strict_score(assigner_list,score_list)

        weighted_score_out=weighted_score(assigner_list,score_list)
        weighted_strict_score_out=weighted_strict_score(assigner_list,score_list)

        output5={'all_score':all_score_out,'all_strict_score':all_strict_score_out,'vital_score':vital_score_out,'vital_strict_score':vital_strict_score_out,'weighted_score':weighted_score_out,'weighted_strict_score':weighted_strict_score_out}

        final_output={'model':model_name,'first_gold_doc_pos':i,'corpus_position':j,'query':query,'generator_output':output1,'nuggetizellm_output':output2,'nuggetizescorellm_output':output3,'nuggetizeassignerllm_output':output4,'scores':output5}


        with open(output_path, "a") as f:
            f.write(json.dumps(final_output) + "\n")
            f.flush()
            os.fsync(f.fileno())


    except KeyboardInterrupt:
        raise

    except Exception as e:

        error_data={'first_gold_doc_pos':i,'corpus_position':j,'error':str(e)}

        with open(retry_error_log_path, "a") as f:
            f.write(json.dumps(error_data) + "\n")
            f.flush()
            os.fsync(f.fileno())