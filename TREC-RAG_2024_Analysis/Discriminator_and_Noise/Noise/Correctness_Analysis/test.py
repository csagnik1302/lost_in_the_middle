import json

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','r') as f:
    out=json.load(f)

for i in range(3):
    for j in range(2):
        out['i']+=1
        out['j']+=1

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','w') as f:
    json.dump(out,f)

print(out)