import gzip
import json
import ast

def prompt_creator_nuggetizeassignerllm(nugget_dict,passage,nugget):

    query=nugget_dict['query']

    prompt1="You are NuggetizeAssignerLLM, an intelligent assistant that can label an atomic nugget based on if it is captured by a given passage."
    prompt2='Based on the query and passage, label the nugget either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support.'
    prompt3=f'Search Query: {query}'
    prompt31=f'Passage: {passage}'
    prompt4=f'Nugget: {nugget}'
    prompt5='Only return the labels (support, partial_support, or not_support). Do not explain.'
    prompt6='Label:'
    

    message=[{"role":"system","content":prompt1},
            {"role":"user","content":prompt2+'\n'+prompt3+'\n'+prompt31+'\n'+prompt4+'\n'+prompt5+'\n'+prompt6}]

    return message, query




if __name__=='__main__':


    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json','r') as f:
        nugget_dict=json.load(f)

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_generator.json','r') as f:
        out=json.load(f)
    
    passage=out['generator_llm_output']


    nugget_list=nugget_dict['NuggetizeLLM_output']

    prompt_list=[]

    for i in nugget_list:
        prompt,q=prompt_creator_nuggetizeassignerllm(nugget_dict,passage,i)

        prompt_list.append(prompt)

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_prompt_nuggetizeassignerllm.json','w', encoding='utf-8') as f:
        json.dump(prompt_list,f,indent=2)

    print(prompt)

