import json

def prompt_creator_noise(input_data):

    query=input_data['query']
    doc_list=input_data['doc']
    

    prompt1="This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    prompt3=f'QUESTION: {query}'
    prompt4='CONTEXT DOCUMENTS:'
    
    prompt5=''
    for i in range(len(doc_list)):
        prompt_temp=f"[{i+1}] {{{" ".join(doc_list[i]['title'].split())}}}: {{{" ".join(doc_list[i]['segment'].split())}}}"
        if i==len(doc_list)-1:
            prompt5+=prompt_temp
        else:
            prompt5+=prompt_temp+'\n'
    
    # prompt6='INSTRUCTION: Please give a complete answer to the question. Cite each context document that supports your answer within brackets [] using the IEEE format.'
    ### TO BE USED FOR Query-Aware Contextualization

    message=[{"role":"system","content":prompt1},
            {"role":"user","content":prompt3+'\n\n'+prompt4+'\n\n'+prompt5+'\n\n'+'Output:'}]

    return message, query



if __name__=='__main__':

    from gold_injector import gold_injector_noise

    retr_set=[]

    PATH=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl'

    input_data=gold_injector_noise(PATH,1,0)

    prompt,query=prompt_creator_noise(input_data)


    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_prompt.json','w', encoding='utf-8') as f:
        json.dump(prompt,f,indent=2)

    print(prompt)
    

