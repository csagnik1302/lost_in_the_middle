import json

def prompt_creator(input_data):

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

    from pathlib import Path
    from gold_injector import gold_injector

    generator=Path(__file__).resolve().parent
    correctness_analysis=generator.parent
    discriminator=correctness_analysis.parent
    disc_parent=discriminator.parent
    trec_2024=disc_parent.parent
    non_factoid=trec_2024.parent
    ROOT=non_factoid.parent

    gold_count=3
    app='2A'


    PATH=ROOT/'Non_Factoid_Analysis'/'TREC-RAG_2024_Analysis'/'Discriminator_and_Noise'/'Data'/'bm25'/f'generator_input_data_gold_fixed_{gold_count}_app{app}.jsonl'


    # retr_set=[]

    # input_data=gold_injector(PATH,1,0)

    # prompt,query=prompt_creator(input_data)


    # with open(ROOT/'Non_Factoid_Analysis'/'TREC-RAG_2024_Analysis'/'Discriminator_and_Noise'/'Discriminator'/'Correctness_Analysis'/'misc'/'sample_prompt_generator.json','w', encoding='utf-8') as f:
    #     json.dump(prompt,f,indent=2)
    

    # print(prompt)


    ############################

    from transformers import AutoTokenizer, AutoConfig
    from tqdm import tqdm

    model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

    tokenizer=AutoTokenizer.from_pretrained(model)
    
    # total=0
    # counter=0

    # for j in range(1,6):
    #     for i in tqdm(range(58)):
    #         input_data=gold_injector(PATH, 1, i)
    #         prompt, query=prompt_creator(input_data)

    #         input_ids=tokenizer.apply_chat_template(prompt, tokenize=True)
    #         token_length=len(input_ids['input_ids'])
    #         total+=token_length
    #         counter+=1

    
    # print(total/counter)
    # print(token_length)

    config = AutoConfig.from_pretrained(model)
    print(config.max_position_embeddings)
