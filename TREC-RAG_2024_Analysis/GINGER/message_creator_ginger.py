def message_creator_ginger(instructions,query,passage):

    message=[{"role":"system",
            "content":instructions},

            {"role":"user",
            "content":("Query: What causes the aurora borealis?\n"
                        "Passage: The aurora borealis, or northern lights, occurs when charged particles from the sun " 
                        "collide with gases in Earth's atmosphere. This collision typically happens near the magnetic poles. " 
                        "Earth's magnetic field channels these particles towards the polar regions, creating a spectacular display " 
                        "of light.")},

            {"role":"assistant",
            "content":("The aurora borealis, or northern lights, occurs when <IN>charged particles from the "
                        "sun collide with gases in Earth's atmosphere</IN>. This collision typically happens "
                        "near the magnetic poles. Earth's magnetic field channels these particles towards the "
                        "polar regions, creating a spectacular display of light.")},

            {"role":"user",
            "content":(f"Query: {query}\n"
                        f"Passage: {passage}")}]

    
    return message


if __name__=='__main__':

    import json

#######################################
    with open('/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/schema_and_related_inputs/prompt_system_instructions.txt','r') as f:
        out=f.readlines()

    prompt1=''

    for i in out:
        prompt1+=i
############################################


##########################################
    with open("/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Data/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl","r") as f:
        out=[]
        for i in f:
            out.append(json.loads(i))


    query=out[1]['query']['text']
    passage=out[1]['candidates'][0]['doc']['segment']

    out=message_creator_ginger(prompt1,query,passage)

    print(out)

    with open('/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/samples/sample_prompt.json','w', encoding='utf-8') as f:
        json.dump(out,f,indent=2)