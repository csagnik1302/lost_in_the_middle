import gzip
import json
import os
from pyserini.index.lucene import LuceneIndexer

list1=['0'+str(i) for i in range(10)]

list2=[str(i) for i in range(10,60)]

list1.extend(list2)

indexer = LuceneIndexer(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/indexes/trec_rag_24/')

count=0

for i in list1:
    PATH=f'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_{i}.json.gz'
    
    with gzip.open(PATH,'r') as f:
        for i in f:

            ms_marco_input=json.loads(i)
            data_dict={'id':ms_marco_input['docid'],'contents':ms_marco_input['segment'].replace('\n','')}

            ############ data_dict might need further cleaning

            data_json=json.dumps(data_dict)
            indexer.add_doc_raw(data_json)

    count += 1
    print(f'Partition: {count} Indexing DONE')

indexer.close()