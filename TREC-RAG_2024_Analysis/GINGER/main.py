import json

with open("/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Data/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl","r") as f:
    out=[]
    for i in f:
        out.append(json.loads(i))

# for i in range(len(out[0]['candidates'])):
#     print(out[0]['candidates'][i]['score'])

print(out[0]['query'])
