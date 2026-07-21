from pyserini.search.lucene import LuceneSearcher
import os
import json

query_path=r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/query_rag25.jsonl'
index_path=r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/BM_25_Retrieval/indexes/trec_rag_24'
retrieval_results_path=r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/bm25/Retrieval_Results.jsonl'

query_json_list=[]

with open(query_path,'r') as f:
    for i in f:
        query_json=json.loads(i)
        query_json_list.append(query_json)

query=[i["title"] for i in query_json_list]
query_id=[i["id"] for i in query_json_list]

searcher = LuceneSearcher(index_path)
searcher.set_bm25()    # k1 and v not set (default is running)

for i in range(len(query)):
	query1=query[i]
	query_id1=query_id[i]

	hits = searcher.search(query1,k=500)

	with open(retrieval_results_path,'a') as f:
		temp={}
		temp['query_id']=query_id1
		temp['hits']=[]
		for m in range(len(hits)):
			temp['hits'].append([hits[m].docid,hits[m].score])
		f.write(json.dumps(temp)+'\n')

		f.flush()
		os.fsync(f.fileno())

	print(f'Query: {i} Retrieval Complete')