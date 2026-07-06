import gzip
import json

def prompt_creator(input_data,query_lookup_data):

    query_id=input_data['query_id']
    doc_id_list=input_data['doc_id']
    
    for i in query_lookup_data:
        if i['query_id']==query_id:
            query=i['query'].replace('\n','')
    
    list1=['0'+str(i) for i in range(10)]
    list2=[str(i) for i in range(10,60)]
    list1.extend(list2)

    doc_title_segment_list=[]   

    count=0

    for i in doc_id_list:

        doc_id_location=i[17:19]

        PATH=fr'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_{doc_id_location}.json.gz'
        with gzip.open(PATH,'r') as f:
            found_in_file=False
            for k in f:
                ms_marco_input=json.loads(k)
                ms_marco_input_docid=ms_marco_input['docid']

                if ms_marco_input_docid==i:
                    passage_title=ms_marco_input['title']
                    passage_segment=ms_marco_input['segment']

                    temp={'title':passage_title,'segment':passage_segment}
                    doc_title_segment_list.append(temp)
                    print(f'doc_segment and doc_title found for {count}')
                    count+=1

                    found_in_file=True
                    break


    prompt1="You are a helpful, detailed, and polite AI assistant. Answer the user's question only using the provided context documents. No need to mention references or citations at the end of your response. No need to generate any more information that what is needed."
    prompt3=f'QUESTION: {query}'
    prompt4='CONTEXT DOCUMENTS:'
    
    prompt5=''
    for i in range(len(doc_title_segment_list)):
        prompt_temp=f"[{i+1}] {{{" ".join(doc_title_segment_list[i]['title'].splitlines())}}}: {{{" ".join(doc_title_segment_list[i]['segment'].splitlines())}}}"
        if i==len(doc_title_segment_list)-1:
            prompt5+=prompt_temp
        else:
            prompt5+=prompt_temp+'\n'
    
    prompt6='INSTRUCTION: Please give a complete answer to the question. Cite each context document that supports your answer within brackets [] using the IEEE format.'
    ### TO BE USED FOR Query-Aware Contextualization

    message=[{"role":"system","content":prompt1},
            {"role":"user","content":prompt3+'\n\n'+prompt4+'\n\n'+prompt5+'\n\n'+prompt6+'\n\n'+'Output:'}]

    return message




if __name__=='__main__':

    from gold_injector import gold_injector

    retr_set=[]

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl', 'r') as f:
        for i in f:
            temp=json.loads(i)
            retr_set.append(temp)    

    input_data=gold_injector(retr_set,0)[3]

    query_lookup_data=[]

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/query_rag24.jsonl','r') as f:
        for i in f:
            query_lookup_data_temp=json.loads(i)
            query_lookup_data.append(query_lookup_data_temp)

    prompt=prompt_creator(input_data,query_lookup_data)


    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Misc/sample_prompt.json','w', encoding='utf-8') as f:
        json.dump(prompt,f,indent=2)
    

