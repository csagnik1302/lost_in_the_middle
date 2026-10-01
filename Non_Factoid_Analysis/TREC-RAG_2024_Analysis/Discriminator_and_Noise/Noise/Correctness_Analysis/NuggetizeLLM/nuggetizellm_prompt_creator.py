import gzip
import json

def prompt_creator_nuggetizellm(input_data):

    query = input_data['query']
    doc = input_data['doc_gold']

    prompt1 = (
        "You are NuggetizeLLM, an intelligent assistant that extracts "
        "atomic nuggets of information needed to comprehensively answer "
        "the search query."
    )

    prompt2 = (
        'Extract and update a list of atomic nuggets of information that '
        'best provide the information required for the query. '
        'Use only the provided context and the initial nugget list. '
        'This is an iterative process. '
        'Return only the final list as a JSON object of the form '
        '{"nuggets": ["nugget 1", "nugget 2", ...]}, '
        'with nothing before or after it. '
        'The list MUST contain at least 3 nuggets and at most 30 nuggets. '
        'Do not return only one nugget. '
        'When the context contains multiple distinct factual pieces of '
        'information relevant to the query, represent them as separate '
        'nuggets rather than combining them into a single nugget. '
        'Each nugget should express one distinct atomic piece of information. '
        'Do not omit a relevant factual detail merely to make the list shorter. '
        'Avoid stating exactly the same fact in multiple nuggets. '
        'Order the nuggets in decreasing order of importance. '
        'Do not use numbering or bullets.'
    )

    prompt3 = f'Search Query: {query}'

    prompt4 = 'Context:'

    prompt41 = f'Search Query: {query}'

    prompt42 = 'Initial Nugget List: []'

    prompt43 = 'Initial Nugget List Length: 0'

    prompt44 = (
        'Extract all distinct relevant atomic facts from the provided context. '
        'Ensure that the final list contains at least 2 nuggets whenever the '
        'context contains at least two distinct pieces of relevant information. '
        'Do not explain your answer. '
        'Do not output questions. '
        'Output only the JSON object.'
    )

    prompt45 = 'Updated Nugget List (JSON object):'

    prompt5 = ''

    for i in range(len(doc)):

        prompt_temp = (
            f"[{i+1}] "
            f"{' '.join(doc[i]['segment'].split())}"
        )

        if i == len(doc) - 1:
            prompt5 += prompt_temp
        else:
            prompt5 += prompt_temp + '\n'

    message = [
        {
            "role": "system",
            "content": prompt1
        },
        {
            "role": "user",
            "content": (
                prompt2 + '\n' +
                prompt3 + '\n' +
                prompt4 + '\n' +
                prompt5 + '\n' +
                prompt41 + '\n' +
                prompt42 + '\n' +
                prompt43 + '\n' +
                prompt44 + '\n' +
                prompt45
            )
        }
    ]

    return message, query




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

