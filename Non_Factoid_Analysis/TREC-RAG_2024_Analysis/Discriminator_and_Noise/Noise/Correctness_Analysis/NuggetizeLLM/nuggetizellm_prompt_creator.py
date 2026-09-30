import gzip
import json

def prompt_creator_nuggetizellm(input_data):

    query=input_data['query']
    doc=input_data['doc_gold']


    prompt1="You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
    # MODIFIED: rules now mirror the NuggetList schema/validator exactly (JSON object output, 1-12 words and at most 120 characters
    # per nugget, 1-30 nuggets, no numbering/bullets, no redundancy) so the model's natural output already satisfies the constraints
    prompt2='Update the list of atomic nuggets of information (1-12 words each), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process). Return only the final list of all nuggets as a JSON object of the form {"nuggets": ["nugget 1", "nugget 2", ...]} (even if no updates), with nothing before or after it. Each nugget must be a plain string of 1-12 words (at most 120 characters), with no numbering or bullets. The list must contain at least 1 and at most 30 nuggets (can be less), keeping only the most vital ones. Make sure there is no redundant information: no two nuggets may state the same fact. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.:'
    prompt3=f'Search Query: {query}'
    prompt4=f'Context:'
    prompt41=f'Search Query: {query}'
    # MODIFIED: the old '{ni−1}' / '{len(ni−1)}' were literal placeholder text (not f-strings), so the model saw them verbatim.
    # This is a single pass with no earlier list, so the initial list is empty.
    prompt42='Initial Nugget List: []'
    prompt43='Initial Nugget List Length: 0'
    # MODIFIED: removed the stray '".' at the end of the string; added an explicit "JSON object only" instruction
    prompt44='Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). Output only the JSON object.'
    # MODIFIED: cue the model to answer in the JSON format
    prompt45='Updated Nugget List (JSON object):'

    prompt5=''
    for i in range(len(doc)):
        # MODIFIED: inner quotes changed to single quotes so the f-string also works on Python < 3.12
        prompt_temp=f"[{i+1}] {{{' '.join(doc[i]['segment'].split())}}}"
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

