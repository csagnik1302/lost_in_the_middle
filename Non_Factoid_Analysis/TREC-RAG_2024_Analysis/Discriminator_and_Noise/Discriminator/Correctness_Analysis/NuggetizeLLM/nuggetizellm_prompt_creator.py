import gzip
import json

def prompt_creator_nuggetizellm(input_data):

    query = input_data['query']
    doc = input_data['doc_gold']

    prompt1 = (
        "You are NuggetizeLLM, an intelligent assistant that creates and updates "
        "a list of atomic nuggets containing the information required to answer a query."
    )

    prompt2 = (
        "Create the final list of atomic nuggets using ONLY the provided context "
        "and the initial nugget list. Do not introduce information that is not "
        "supported by the context.\n\n"

        "IMPORTANT RULES:\n"
        "1. Return at least 3 nuggets whenever the context contains 3 or more "
        "distinct relevant facts.\n"
        "2. Return at most 30 nuggets.\n"
        "3. Each nugget must express one distinct factual piece of information.\n"
        "4. Do NOT combine multiple distinct facts into a single nugget merely "
        "to reduce the number of nuggets.\n"
        "5. Do NOT omit relevant factual information from the context just to "
        "make the list shorter.\n"
        "6. Do NOT invent, infer, or hallucinate facts that are not supported "
        "by the provided context.\n"
        "7. Do NOT produce duplicate nuggets or multiple nuggets stating exactly "
        "the same fact.\n"
        "8. Nuggets may be short or long when necessary to preserve the complete "
        "meaning of the factual information.\n"
        "9. Do NOT use numbering, bullet points, commentary, explanations, "
        "headings, or introductory/concluding text inside the nuggets.\n"
        "10. Return ONLY the final JSON object.\n\n"

        "The required output format is exactly:\n"
        '{"nuggets": ["nugget 1", "nugget 2", "nugget 3", ....]}\n\n'

        "If the context contains several distinct relevant facts, preserve them "
        "as separate nuggets rather than collapsing them into one.\n"
        "If an initial nugget list is provided, retain its useful information "
        "and update or add nuggets when the context provides additional relevant "
        "information.\n"
        "If an initial nugget is contradicted by the context, update it using "
        "the information supported by the context."
    )

    prompt3 = f"Search Query: {query}"

    prompt4 = "Context:"

    prompt5 = ""

    for i in range(len(doc)):
        prompt_temp = f"[{i+1}] {{{' '.join(doc[i]['segment'].split())}}}"

        if i == len(doc) - 1:
            prompt5 += prompt_temp
        else:
            prompt5 += prompt_temp + "\n"

    prompt6 = (
        "Initial Nugget List: []\n"
        "Initial Nugget List Length: 0\n\n"
        "Before producing the answer, identify all distinct relevant factual "
        "information needed to answer the query. Represent those facts as "
        "separate atomic nuggets.\n\n"
        "Do not explain your reasoning.\n"
        "Do not output anything except the JSON object."
    )

    message = [
        {
            "role": "system",
            "content": prompt1
        },
        {
            "role": "user",
            "content": (
                prompt2
                + "\n\n"
                + prompt3
                + "\n\n"
                + prompt4
                + "\n"
                + prompt5
                + "\n\n"
                + prompt3
                + "\n"
                + prompt6
                + "\n\n"
                + "Updated Nugget List (JSON object):"
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


    prompt,query=prompt_creator_nuggetizellm(input_data)


    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_prompt_nuggetizellm.json','w', encoding='utf-8') as f:
        json.dump(prompt,f,indent=2)


    print(prompt)