# import json
# import gzip

# list1=['0'+str(i) for i in range(10)]
# list2=[str(i) for i in range(10,60)]
# list1.extend(list2)


# for i in list1:
#     PATH=f'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_{i}.json.gz'
    
#     with gzip.open(PATH,'r') as f:
#         for i in f:

#             ms_marco_input=json.loads(i)
#             print(ms_marco_input)
#             break
#     break

# a=2

# print(f'{{{a}}}')

import json

retr_set=[]

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_1.jsonl','r') as f:
    for i in f:
            temp=json.loads(i)
            retr_set.append(temp)


retr_set1=[]

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_3.jsonl','r') as f:
    for i in f:
            temp=json.loads(i)
            retr_set1.append(temp)



query_match_list=[i['query_id'] for i in retr_set1]

generator_input_dict=[]
generator_input_dict1=[]

count=0

for i in retr_set:

    generator_input_dict_temp={}
    generator_input_dict_temp['doc_id_discriminator']=[]
    generator_input_dict_temp['doc_id_gold']=[]

    if i['query_id'] in query_match_list:
        generator_input_dict_temp['query_id']=i['query_id']
        for j in retr_set1:
            if j['query_id']==i['query_id']:
                gold_len=len(j['doc_id'])
                gold_list=j['doc_id']
        if len(i['doc_id'])>=57 and gold_len>=3:
            count_disc=0
            count_gold=0
            for k in i['doc_id']:
                generator_input_dict_temp['doc_id_discriminator'].append(k)
                count_disc+=1
                if count_disc==57:
                    break
            for l in gold_list:
                generator_input_dict_temp['doc_id_gold'].append(l)
                count_gold+=1
                if count_gold==3:
                    break
        

    generator_input_dict1.append(generator_input_dict_temp)
    generator_input_dict=[i for i in generator_input_dict1 if len(i['doc_id_discriminator'])==57 and len(i['doc_id_gold'])==3]

method='bm25'
            
with open(rf'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/{method}/generator_input_data_id_gold_fixed_3_without_rag.jsonl','w') as f:
        for i in generator_input_dict:
            f.write(json.dumps(i)+'\n')
