import gzip
import json
import os
from tqdm import tqdm

def id_to_content(input_data,query_lookup_data):

    query_id=input_data['query_id']
    doc_id_discriminator_list=input_data['doc_id_discriminator']
    doc_id_gold_list=input_data['doc_id_gold']
    
    for i in query_lookup_data:
        if i['id']==query_id:
            query=i['title'].replace('\n','')
    
    list1=['0'+str(i) for i in range(10)]
    list2=[str(i) for i in range(10,60)]
    list1.extend(list2)

    doc_title_segment_discriminator_list=[]
    doc_title_segment_gold_list=[]     


    for i in doc_id_discriminator_list:

        doc_id_location=i[17:19]

        PATH=fr'/home/irlab/sagnik/Non_Factoid_Analysis/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_{doc_id_location}.json.gz'
        with gzip.open(PATH,'r') as f:
            found_in_file=False
            for k in f:
                ms_marco_input=json.loads(k)
                ms_marco_input_docid=ms_marco_input['docid']

                if ms_marco_input_docid==i:
                    passage_title=ms_marco_input['title']
                    passage_segment=ms_marco_input['segment']

                    temp={'title':passage_title,'segment':passage_segment}
                    doc_title_segment_discriminator_list.append(temp)
                    break

        
    for i in doc_id_gold_list:

        doc_id_location=i[17:19]

        PATH=fr'/home/irlab/sagnik/Non_Factoid_Analysis/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_{doc_id_location}.json.gz'
        with gzip.open(PATH,'r') as f:
            found_in_file=False
            for k in f:
                ms_marco_input=json.loads(k)
                ms_marco_input_docid=ms_marco_input['docid']

                if ms_marco_input_docid==i:
                    passage_title=ms_marco_input['title']
                    passage_segment=ms_marco_input['segment']

                    temp={'title':passage_title,'segment':passage_segment}
                    doc_title_segment_gold_list.append(temp)
                    break


    
    output={"query":query,"doc_discriminator":doc_title_segment_discriminator_list,"doc_gold":doc_title_segment_gold_list}

    return output


if __name__=="__main__":

    retr_set=[]
    query_lookup_data=[]
    method='bm25'

    with open(fr'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/{method}/generator_input_data_id_gold_fixed_3_without_rag.jsonl', 'r') as f:
        for i in f:
            temp=json.loads(i)
            retr_set.append(temp)


    with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/query_rag25.jsonl','r') as f:
        for i in f:
            query_lookup_data_temp=json.loads(i)
            query_lookup_data.append(query_lookup_data_temp)   



    for i in tqdm(retr_set):

        output=id_to_content(i,query_lookup_data)

        with open(fr"/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2025_Analysis/Discriminator_and_Noise/Data/{method}/generator_input_data_gold_fixed_3_without_rag.jsonl", "a") as f:
            f.write(json.dumps(output) + "\n")
            f.flush()
            os.fsync(f.fileno()) 

