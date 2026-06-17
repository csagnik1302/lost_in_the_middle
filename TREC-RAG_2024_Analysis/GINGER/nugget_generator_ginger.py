from transformers import AutoTokenizer, AutoModelForCausalLM


def nugget_generator_ginger(prompt,model_name,hf_token_key):

    model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,token=hf_token_key)
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,token=hf_token_key)

    tokenized_input=tokenizer(prompt,return_tensors='pt')
    input_size=tokenized_input['input_ids'].shape[1]

    model_output=model.generate(tokenized_input['input_ids'],tokenizer=tokenizer)
    new_output=model_output[0][input_size:]

    decoded_output=tokenizer.decode(new_output,skip_special_tokens=True)

    return decoded_output


if __name__=='__main__':

    import json
    from prompt_creator_ginger import prompt_creator_ginger


    with open('/home/irlab/sagnik/API_KEY','r') as f:
        hf_token_key=f.read()


    ##########################################
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

    prompt=prompt_creator_ginger(prompt1,query,passage)
    #############################################

    model='meta-llama/Meta-Llama-3.1-8B-Instruct'

    output=nugget_generator_ginger(prompt,model,hf_token_key)

    print(output)
    
    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/samples/sample_nugget_tagged_output.txt','w') as f:
        f.write(output)
