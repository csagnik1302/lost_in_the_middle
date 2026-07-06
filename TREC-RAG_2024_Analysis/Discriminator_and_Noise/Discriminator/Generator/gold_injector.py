def gold_injector(corpus,position_first):

    output=[]
    for i in corpus:
        doc_id=i['doc_id_discriminator']
        doc_id_gold=i['doc_id_gold']
        query_id=i['query_id']

        output_temp={'query_id':query_id}

        for i in range(len(doc_id_gold)):
            doc_id.insert(position_first+i,doc_id_gold[i])

        output_temp['doc_id']=(doc_id)

        output.append(output_temp)
    
    return output




if __name__=='__main__':

    import json

    retr_set=[]

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl', 'r') as f:
        for i in f:
            temp=json.loads(i)
            retr_set.append(temp)    

    out=gold_injector(retr_set,1)

    print(out[0])



