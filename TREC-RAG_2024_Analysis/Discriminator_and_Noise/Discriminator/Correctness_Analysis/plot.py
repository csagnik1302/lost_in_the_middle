import json
import matplotlib.pyplot as plt

output=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_output.jsonl','r') as f:
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
    ("All Score", all_score),
    ("All Strict Score", all_strict_score),
    ("Vital Score", vital_score),
    ("Vital Strict Score", vital_strict_score),
    ("Weighted Score", weighted_score),
    ("Weighted Strict Score", weighted_strict_score),
]

output_dir=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/Plots'

for title, scores in plots:
    plt.figure(figsize=(8, 5))
    plt.plot(gold_pos, scores, marker='o')
    plt.title(title)
    plt.xlabel("First Gold Document Position")
    plt.ylabel(title)
    plt.xticks(gold_pos)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()