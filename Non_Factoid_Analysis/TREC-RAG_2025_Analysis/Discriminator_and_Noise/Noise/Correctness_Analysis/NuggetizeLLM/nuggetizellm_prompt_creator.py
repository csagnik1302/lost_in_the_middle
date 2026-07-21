import gzip
import json

def prompt_creator_nuggetizellm_noise(input_data):

    query=input_data['query']
    doc=input_data['doc_gold']


    prompt1="You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
    prompt2='Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process). Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most 30 nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.:'
    prompt3=f'Search Query: {query}'
    prompt4=f'Context:'
    prompt41=f'Search Query: {query}'
    prompt42='Initial Nugget List: {ni−1}'
    prompt43='Initial Nugget List Length: {len(ni−1)}'
    prompt44='Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions)".'
    prompt45='Updated Nugget List:'

    prompt5=''
    for i in range(len(doc)):
        prompt_temp=f"[{i+1}] {{{" ".join(doc[i]["segment"].split())}}}"
        if i==len(doc)-1:
            prompt5+=prompt_temp
        else:
            prompt5+=prompt_temp+'\n'
    

    message=[{"role":"system","content":prompt1},
            {"role":"user","content":prompt2+'\n'+prompt3+'\n'+prompt4+'\n'+prompt5+'\n'+prompt41+'\n'+prompt42+'\n'+prompt43+'\n'+prompt44+'\n'+prompt45}]

    return message,query




if __name__=='__main__':

    retr_set=[]

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/generator_input_data_gold_fixed_3.jsonl', 'r') as f:
        for i in f:
            temp=json.loads(i)
            retr_set.append(temp)    

    input_data=retr_set[0]


    prompt,query=prompt_creator_nuggetizellm_noise(input_data)


    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_prompt_nuggetizellm.json','w', encoding='utf-8') as f:
        json.dump(prompt,f,indent=2)


    print(prompt)

