import json

method='bm25'

retr_set=[]
qrel_0=[]
qrel_1=[]
qrel_2=[]
qrel_3=[]


with open(rf'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/{method}/Retrieval_Results.jsonl', 'r') as f:
    for i in f:
        temp=json.loads(i)
        retr_set.append(temp)


with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_0.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_0.append(temp)


with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_1.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_1.append(temp)

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_2.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_2.append(temp)

with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/qrels/qrels_3.jsonl','r') as f:
    for i in f:
        temp=json.loads(i)
        qrel_3.append(temp)




ignore_query1=[i['query_id'] for i in qrel_1 if len(i['doc_id'])<57]
ignore_query2=[i['query_id'] for i in qrel_2 if len(i['doc_id'])<57]

ignore_query_comb=[i for i in ignore_query1 if i in ignore_query2]

ignore_query=[]



for i in ignore_query_comb:
    m=0
    for j in qrel_1:
        if j['query_id']==i:
            m+=len(j['doc_id'])
        
            for k in qrel_2:
                if k['query_id']==i:
                    m+=len(k['doc_id'])

                    if m<57:
                        ignore_query.append(i)




generator_input_dict=[]

test_count=0
test_count1=0

for i in retr_set:

    if i['query_id'] in ignore_query:
        continue
    
    generator_input_dict_temp={}
    generator_input_dict_temp['doc_id_discriminator']=[]
    generator_input_dict_temp['doc_id_gold']=[]

    query_id=i['query_id']
    hits=i['hits']

    generator_input_dict_temp['query_id']=query_id


        
    for j in qrel_1:

        if j['query_id']==query_id:
            count=0
            for k in range(len(hits)):
                if hits[k][0] in j['doc_id']:
                    generator_input_dict_temp['doc_id_discriminator'].append(hits[k][0])
                    count+=1

                    if count==57:
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


    current_disc_count=len(generator_input_dict_temp['doc_id_discriminator'])

    if current_disc_count<57:

        counter=current_disc_count

        for j in qrel_2:

            flag=False

            if j['query_id']==query_id:
                for k in range(len(hits)):
                    if hits[k][0] in j['doc_id']:
                        if hits[k][0] not in generator_input_dict_temp['doc_id_discriminator']:
                            generator_input_dict_temp['doc_id_discriminator'].append(hits[k][0])
                            counter+=1

                            if counter==57:
                                flag=True
                                break

                if flag==True:
                    break
                        


    print(f'For Query: {test_count} | Discriminator Count: {len(generator_input_dict_temp['doc_id_discriminator'])} | Gold Count: {len(generator_input_dict_temp['doc_id_gold'])}')
    # test_count+=1


    generator_input_dict.append(generator_input_dict_temp)


# ################################# HAS IN SOME CASES, A GOLD COUNT OF LESS THAN 3, TO PREVENT QUALITY DILUTION, WE ARE NOT USING IT FOR NOW ################
# with open(rf'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/{method}/generator_input_data.jsonl','w') as f:
#         for i in generator_input_dict:
#             f.write(json.dumps(i)+'\n')

# ####################################################################################################################################

generator_input_dict_gold_fixed_3=[i for i in generator_input_dict if len(i['doc_id_gold'])==3 and len(i['doc_id_discriminator'])==57]

with open(rf'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/{method}/generator_input_data_id_gold_fixed_3.jsonl','w') as f:
        for i in generator_input_dict_gold_fixed_3:
            f.write(json.dumps(i)+'\n')



            

