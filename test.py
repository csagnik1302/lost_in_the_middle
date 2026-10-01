import json

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/bm25/pipeline_output_gold_3_Qwen2.5-7B-Instruct-bnb-4bit_app2A.jsonl', 'r') as f:
    inp = [json.loads(line) for line in f]

flag=True
m=[]
counter=1

for i in inp:
    if len(i['nuggetizellm_output'])==1:
        flag=False
        m.append(counter)
        break
    counter+=1

print(flag, m)