import json
import matplotlib.pyplot as plt

output=[]

model_name='mistral-7b-instruct-v0.3-bnb-4bit'
method='bm25'

with open(rf'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/{method}/pipeline_output_{model_name}.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        output.append(temp)


gold_pos=[]

for i in output:
    gold_temp=i['first_gold_doc_pos']
    
    if gold_temp not in gold_pos:
        gold_pos.append(gold_temp)

all_score=[]
all_strict_score=[]
vital_score=[]
vital_strict_score=[]
weighted_score=[]
weighted_strict_score=[]


for i in gold_pos:
    all_score_temp=0
    all_strict_score_temp=0
    vital_score_temp=0
    vital_strict_score_temp=0
    weighted_score_temp=0
    weighted_strict_score_temp=0
    count=0
    for j in output:
        if j['first_gold_doc_pos']==i:
            count+=1
            all_score_temp+=j['scores']['all_score']
            all_strict_score_temp+=j['scores']['all_strict_score']
            vital_score_temp+=j['scores']['vital_score']
            vital_strict_score_temp+=j['scores']['vital_strict_score']
            weighted_score_temp+=j['scores']['weighted_score']
            weighted_strict_score_temp+=j['scores']['weighted_strict_score']

    all_score.append(all_score_temp/count)
    all_strict_score.append(all_strict_score_temp/count)
    vital_score.append(vital_score_temp/count)
    vital_strict_score.append(vital_strict_score_temp/count)
    weighted_score.append(weighted_score_temp/count)
    weighted_strict_score.append(weighted_strict_score_temp/count)


plots = [
    (f"All Score (Model: {model_name})", all_score),
    (f"All Strict Score (Model: {model_name})", all_strict_score),
    (f"Vital Score (Model: {model_name})", vital_score),
    (f"Vital Strict Score (Model: {model_name})", vital_strict_score),
    (f"Weighted Score (Model: {model_name})", weighted_score),
    (f"Weighted Strict Score (Model: {model_name})", weighted_strict_score),
]

output_dir=rf'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/Plots/{method}'

for title, scores in plots:
    plt.figure(figsize=(8, 5))
    plt.plot(gold_pos, scores, marker='o')
    plt.title(title)
    plt.xlabel("First Gold Document Position")
    plt.ylabel(title.replace(f'(Model: {model_name})',''))
    plt.xticks(gold_pos)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()