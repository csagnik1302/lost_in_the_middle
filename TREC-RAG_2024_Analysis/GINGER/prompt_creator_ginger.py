def prompt_creator_ginger(instructions,query,passage):
    
    prompt=instructions+'\n\n'+f'Question: {query}'+' '+f'Passage: {passage}'
    
    return prompt


if __name__=='__main__':

    import json

    with open('/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/prompt_schema.txt','r') as f:
        out=f.readlines()

    prompt1=''

    for i in out:
        prompt1+=i




    with open("/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Data/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl","r") as f:
        out=[]
        for i in f:
            out.append(json.loads(i))


    query=out[0]['query']['text']
    passage=out[0]['candidates'][0]['doc']['segment']

    out=prompt_creator_ginger(prompt1,query,passage)

    print(out)

    with open('/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/samples/sample_prompt.txt','w') as f:
        f.writelines(out)