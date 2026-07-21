import json

retr_set=[]

with open(rf'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/bm25/Retrieval_Results.jsonl', 'r') as f:
    for i in f:
        temp=json.loads(i)
        print(len(temp['hits']))
