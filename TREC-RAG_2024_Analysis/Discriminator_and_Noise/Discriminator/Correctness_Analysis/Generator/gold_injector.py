import json

def gold_injector(input_path,position_first_gold_index,corpus_index):

    corpus=[]

    with open(input_path,'r') as f:
        for i in f:
            corpus_temp=json.loads(i)
            corpus.append(corpus_temp)  

    input=corpus[corpus_index]

    output=[]

    doc_discriminator=input['doc_discriminator']
    doc_gold=input['doc_gold']
    query=input['query']

    output={'query':query}

    for i in range(len(doc_gold)):
        doc_discriminator.insert(position_first_gold_index+i,doc_gold[i])

    output['doc']=(doc_discriminator)
    
    return output




if __name__=='__main__':

    retr_set=[]

    PATH='/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl' 

    out=gold_injector(PATH,1,0)

    print(out)

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/gold_injector_output.json','w', encoding='utf-8') as f:
        json.dump(out,f,indent=2)

