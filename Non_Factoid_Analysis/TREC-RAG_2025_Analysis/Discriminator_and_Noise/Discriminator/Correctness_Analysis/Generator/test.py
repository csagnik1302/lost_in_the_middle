from transformers import AutoTokenizer, AutoModelForCausalLM
from gold_injector import gold_injector
from prompt_creator import prompt_creator
import json
import gzip
import torch

#####################
with open(r'C:\lost-in-the-middle\API_KEY','r') as f:
    hf_token=f.read()


model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"


model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation="flash_attention_2")
tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

retr_set=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl', 'r') as f:
    for i in f:
        temp=json.loads(i)
        retr_set.append(temp)    

input_data=gold_injector(retr_set,1,0)

query_lookup_data=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/query_rag24.jsonl','r') as f:
    for i in f:
        query_lookup_data_temp=json.loads(i)
        query_lookup_data.append(query_lookup_data_temp)

prompt=prompt_creator(input_data,query_lookup_data)

input_processed=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)
inputs=tokenizer(input_processed,return_tensors="pt").to(model.device)
input_size=inputs['input_ids'].shape[1]

print(input_size)

# # # from pathlib import Path
# # # import gzip
# # # import shutil

# # # # Folder containing the JSON files
# # # folder = Path(r"C:\lost-in-the-middle\TREC-RAG_2024_Analysis\Discriminator_and_Noise\Discriminator\Data\msmarco_v2.1_doc_segmented")

# # # for json_file in folder.glob("*.json"):
# # #     gzip_file = json_file.with_suffix(json_file.suffix + ".gz")

# # #     with open(json_file, "rb") as f_in:
# # #         with gzip.open(gzip_file, "wb") as f_out:
# # #             shutil.copyfileobj(f_in, f_out)

# # #     print(f"Compressed: {json_file.name} -> {gzip_file.name}")

# # #     json_file.unlink()

# # import re

# # headings = [
# #     "references",
# #     "reference",
# #     "sources",
# #     "source",
# #     "citations",
# #     "citation",
# #     "bibliography",
# #     "works cited",
# # ]

# # with open(r'C:\lost-in-the-middle\TREC-RAG_2024_Analysis\Discriminator_and_Noise\Discriminator\Misc\sample_output.txt','r') as f:
# #     out=f.readlines()

# # text=''

# # for i in out:
# #     text+=i


# # pattern = (
# #     r"(?is)"
# #     r"\n+\s*(?:" + "|".join(re.escape(h) for h in headings) + r")\s*:?.*$"
# # )

# # text = re.sub(pattern, "", text).rstrip()
# # text = text.replace("\n", " ")
# # print(text)

# import sys

# sys.path.insert(0,r'C:\lost-in-the-middle\TREC-RAG_2024_Analysis\Discriminator_and_Noise\Discriminator\Correctness_Analysis\Evaluation')

# from evaluator import all_score

