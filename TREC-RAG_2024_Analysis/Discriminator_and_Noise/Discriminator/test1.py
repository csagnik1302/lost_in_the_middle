from pyserini.search.lucene import LuceneSearcher

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/topics.rag24.test.txt','r') as f:
    query=f.readlines()

query1=query[1][11:]

searcher = LuceneSearcher(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/indexes/trec_rag_24')
hits = searcher.search(query1)

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')