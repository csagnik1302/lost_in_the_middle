import json

retr_set=[]
qrel_0=[]
qrel_1=[]
qrel_2=[]
qrel_3=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/Retrieval Results.jsonl', 'r') as f:
    for i in f:
        temp=json.loads(i)
        retr_set.append(temp)


with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_0.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_0.append(temp)


with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_1.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_1.append(temp)

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_2.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_2.append(temp)

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/qrels/qrels_3.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_3.append(temp)



generator_input_dict=[]

test_count=0
test_count1=0

for i in retr_set:
    generator_input_dict_temp={}
    generator_input_dict_temp['doc_id_discriminator']=[]
    generator_input_dict_temp['doc_id_gold']=[]

    query_id=i['query_id']
    hits=i['hits']

    generator_input_dict_temp['query_id']=query_id

    for j in qrel_0:
        if j['query_id']==query_id:
            count=0
            for k in range(len(hits)):
                if hits[k][0] in j['doc_id']:
                    generator_input_dict_temp['doc_id_discriminator'].append(hits[k][0])
                    count+=1

                    if count==57:
                        print(f'All 57 discriminators found for query: {test_count}')
                        test_count+=1
                        break

            break


    for l in qrel_3:
        if l['query_id']==query_id:
            count1=0
            for m in range(len(hits)):
                if hits[m][0] in l['doc_id']:
                    generator_input_dict_temp['doc_id_gold'].append(hits[m][0])
                    count1+=1

                    if count1==3:
                        break

            break


    generator_input_dict.append(generator_input_dict_temp)



with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data.jsonl','a') as f:
        for i in generator_input_dict:
            f.write(json.dumps(i)+'\n')

        


            

