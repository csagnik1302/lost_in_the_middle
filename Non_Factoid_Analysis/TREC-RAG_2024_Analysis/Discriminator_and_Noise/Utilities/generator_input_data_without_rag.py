import json

method='bm25'

retr_set=[]
qrel_0=[]
qrel_1=[]
qrel_2=[]
qrel_3=[]


with open(rf'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/{method}/Retrieval_Results.jsonl', 'r') as f:
    for i in f:
        temp=json.loads(i)
        retr_set.append(temp)


with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/qrels/qrels_0.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_0.append(temp)


with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/qrels/qrels_1.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_1.append(temp)

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/qrels/qrels_2.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_2.append(temp)

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/qrels/qrels_3.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_3.append(temp)




# ignore_query1=[i['query_id'] for i in qrel_1 if len(i['doc_id'])<57]
# ignore_query0=[i['query_id'] for i in qrel_0]

# ignore_query_comb=[i for i in ignore_query1 if i in ignore_query0]

# ignore_query=[]



# for i in ignore_query_comb:
#     m=0
#     for j in qrel_1:
#         if j['query_id']==i:
#             m+=len(j['doc_id'])
        
#             for k in qrel_0:
#                 if k['query_id']==i:
#                     m+=len(k['doc_id'])

#                     if m<57:
#                         ignore_query.append(i)



query_id_set=[]

for i in qrel_1:
    query_id_set.append(i['query_id'])

output=[]

for i in query_id_set:

    for j in qrel_1:
        if j['query_id']==i:
            disc_set=j['doc_id']
    
    for j in qrel_3:
        if j['query_id']==i:
            gold_set=j['doc_id']
    
    if len(disc_set)<57:
        continue

    if len(gold_set)<3:
        continue

    output_temp={'doc_id_discriminator':[], 'doc_id_gold':[], 'query_id':i}

    disc_counter=0
    while len(output_temp['doc_id_discriminator'])<57:
        output_temp['doc_id_discriminator'].append(disc_set[disc_counter])
        disc_counter+=1

    gold_counter=0
    while len(output_temp['doc_id_gold'])<3:
        output_temp['doc_id_gold'].append(gold_set[gold_counter])
        gold_counter+=1  

    output.append(output_temp)
                

with open(rf'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/{method}/generator_input_data_id_gold_fixed_3_without_rag.jsonl','w') as f:
        for i in output:
            f.write(json.dumps(i)+'\n')