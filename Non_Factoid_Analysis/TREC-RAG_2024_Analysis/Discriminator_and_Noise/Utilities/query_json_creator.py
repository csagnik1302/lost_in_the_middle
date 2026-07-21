import json

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Misc/topics.rag24.test.txt','r') as f:
    k=f.readlines()

out=[]

for i in k:
    temp={}
    for j in range(len(i)):
        if i[j].isnumeric()==False and i[j]!='-':
            temp['query_id']=i[:j]
            break
    for j in range(len(i)):
        if i[j].isalpha()==True:
            temp['query']=i[j:]
            break

    out.append(temp)
    
    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/query_rag24.jsonl','w') as f:
        for i in out:
            f.write(json.dumps(i)+'\n')

