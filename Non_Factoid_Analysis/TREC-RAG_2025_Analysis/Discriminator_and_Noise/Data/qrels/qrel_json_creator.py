import json

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/2025-rag-qrels.txt','r') as f:
    qrel_data=f.readlines()

qrel_data_split=[]

for i in range(len(qrel_data)):
    temp=qrel_data[i].split(' ')
    qrel_data_split.append(temp)

out_0=[]
out_1=[]
out_2=[]
out_3=[]

for i in qrel_data_split:
    temp={}
    temp['query_id']=i[0]
    temp['doc_id']=i[2]
    if int(i[-1][-2])==0:
        out_0.append(temp)
    elif int(i[-1][-2])==1:
        out_1.append(temp)
    elif int(i[-1][-2])==2:
        out_2.append(temp)
    elif int(i[-1][-2])==3:
        out_3.append(temp)


result_out_0 = {}
result_out_1 = {}
result_out_2 = {}
result_out_3 = {}


for item in out_0:
    query_id = item["query_id"]
    doc_id = item["doc_id"]

    if query_id not in result_out_0:
        result_out_0[query_id] = []

    result_out_0[query_id].append(doc_id)


for item in out_1:
    query_id = item["query_id"]
    doc_id = item["doc_id"]

    if query_id not in result_out_1:
        result_out_1[query_id] = []

    result_out_1[query_id].append(doc_id)


for item in out_2:
    query_id = item["query_id"]
    doc_id = item["doc_id"]

    if query_id not in result_out_2:
        result_out_2[query_id] = []

    result_out_2[query_id].append(doc_id)


for item in out_3:
    query_id = item["query_id"]
    doc_id = item["doc_id"]

    if query_id not in result_out_3:
        result_out_3[query_id] = []

    result_out_3[query_id].append(doc_id)


with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_0.jsonl','w') as f:
    for i,j in result_out_0.items():
        temp={}
        temp['query_id']=i
        temp['doc_id']=j
        f.write(json.dumps(temp)+'\n')

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_1.jsonl','w') as f:
    for i,j in result_out_1.items():
        temp={}
        temp['query_id']=i
        temp['doc_id']=j
        f.write(json.dumps(temp)+'\n')

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_2.jsonl','w') as f:
    for i,j in result_out_2.items():
        temp={}
        temp['query_id']=i
        temp['doc_id']=j
        f.write(json.dumps(temp)+'\n')
    
with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_3.jsonl','w') as f:
    for i,j in result_out_3.items():
        temp={}
        temp['query_id']=i
        temp['doc_id']=j
        f.write(json.dumps(temp)+'\n')