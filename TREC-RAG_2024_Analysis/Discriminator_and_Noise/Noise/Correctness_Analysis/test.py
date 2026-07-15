# # import json

# # with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','r') as f:
# #     out=json.load(f)

# # for i in range(3):
# #     for j in range(2):
# #         out['i']+=1
# #         out['j']+=1

# # with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','w') as f:
# #     json.dump(out,f)

# # print(out)


# try:
#     l={"hi":"Hello"My name""}
# except Exception as e:
#     print('Error')

import json
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

input=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_error_log_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
    for i in f:
        input.append(json.loads(i))

indices_missed=[]

for i in input:
    indices_missed.append([i['first_gold_doc_pos'],i['corpus_position']])

print(indices_missed)