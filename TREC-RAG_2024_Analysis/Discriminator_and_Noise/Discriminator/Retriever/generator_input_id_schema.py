import json


####### PATHS #############

retriever_output=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/Retrieval Results.jsonl'

qrels_0=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_0.jsonl'
qrels_1=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_1.jsonl'
qrels_2=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_2.jsonl'
qrels_3=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_3.jsonl'

###############################################

ret_data=[]
qrel_0=[]
qrel_1=[]
qrel_2=[]
qrel_3=[]

with open(retriever_output,'r') as f:
    for i in f:
        temp=json.loads(i)
        ret_data.append(temp)


with open(qrels_0,'r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_0.append(temp)


with open(qrels_1,'r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_1.append(temp)

with open(qrels_2,'r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_2.append(temp)

with open(qrels_3,'r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_3.append(temp)

# print(ret_data[0])
# print('############')
# print(qrel_0[0])
# print('############')
# print(qrel_1[0])
# print('############')
# print(qrel_2[0])
# print('############')
# print(qrel_3[0])

out=[]

for i in ret_data:
    temp={}
    temp['query_id']=i['query_id']
    temp['doc_id']=[]

    retrieval_list_1=i['hits']
    retrieval_list=[i[0] for i in retrieval_list_1]

print(retrieval_list)




