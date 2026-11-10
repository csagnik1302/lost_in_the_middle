# 


##########################
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
import numpy as np
from math import floor
from dotenv import load_dotenv

setproctitle("Discriminator")

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
from dotenv import load_dotenv

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

retr_set_path=rf'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/{method}/generator_input_data_gold_fixed_{gold_count}_app2A.jsonl'

autonugg_model=AutoModelForCausalLM.from_pretrained(autonugg_model_name,token=hf_token,attn_implementation='flash_attention_2')
model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
autonugg_tokenizer=AutoTokenizer.from_pretrained(autonugg_model_name,token=hf_token)
tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)
