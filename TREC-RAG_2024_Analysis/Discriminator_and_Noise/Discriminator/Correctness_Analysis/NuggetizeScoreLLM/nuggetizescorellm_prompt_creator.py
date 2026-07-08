import gzip
import json
import ast

def prompt_creator_nuggetizescorellm(nugget_dict,nugget):

    query=nugget_dict['query']

    prompt1="You are NuggetizeScoreLLM, an intelligent assistant that can label an atomic nugget based on their importance for a given search query."
    prompt2='Based on the search query, label the nugget either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential.'
    prompt3=f'Search Query: {query}'
    prompt4=f'Nugget: {nugget}'
    prompt5='Only return the labels (vital or okay). Do not explain.'
    prompt6='Label:'
    

    message=[{"role":"system","content":prompt1},
            {"role":"user","content":prompt2+'\n'+prompt3+'\n'+prompt4+'\n'+prompt5+'\n'+prompt6}]

    return message, query




if __name__=='__main__':


    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json','r') as f:
        nugget_dict=json.load(f)


    nugget_list_temp=nugget_dict['NuggetizeLLM_output']

    for i in range(len(nugget_list_temp)):
        if nugget_list_temp[i]=='[':
            starting_ind=i
        if nugget_list_temp[i]==']':
            ending_ind=i

    nugget_list=ast.literal_eval(nugget_list_temp[starting_ind:ending_ind+1])

    prompt_list=[]

    for i in nugget_list:
        prompt,q=prompt_creator_nuggetizescorellm(nugget_dict,i)

        prompt_list.append(prompt)

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_prompt_nuggetizescorellm.json','a', encoding='utf-8') as f:
        json.dump(prompt_list,f,indent=2)

    print(nugget_dict)

