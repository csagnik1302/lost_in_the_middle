import json

with open("/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Data/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl","r") as f:
    data=[]
    for i in f:
        data.append(json.loads(i))

query=data[0]['query']['text']

# with open(r"/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Pipeline_outputs/annotated_passages",'r') as f:
#     test=f.readlines()

# count=0
# for i in range(5):
passage_ground=data[0]['candidates'][1]
#     passage_gen=test[i][4:-5]
#     if passage_gen in passage_ground:
#         count+=1

# print(count)


# nugget_dict={'hi':[1,2,3,4],'bye':[5,4,3,6]}

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Pipeline_outputs/nuggets.jsonl','w') as f:
#     for i,j in nugget_dict.items():
#         temp={}
#         temp[i]=j
#         json.dump(temp,f)
#         f.write('\n')


print(passage_ground)