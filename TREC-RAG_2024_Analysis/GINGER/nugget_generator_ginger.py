from transformers import AutoTokenizer, AutoModelForCausalLM


def nugget_generator_ginger(message,model_name,hf_token_key,source_passage):

    model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,token=hf_token_key)
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,token=hf_token_key)

    message_processed=tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)

    tokenized_input=tokenizer(message_processed,return_tensors='pt')
    input_size=tokenized_input['input_ids'].shape[1]    

    tokenized_source_passage=tokenizer(source_passage,return_tensors='pt')
    output_min_size=tokenized_source_passage['input_ids'].shape[1]

    model_output=model.generate(tokenized_input['input_ids'],tokenizer=tokenizer,min_new_tokens=output_min_size, max_length=1000)
    new_output=model_output[0][input_size:]

    decoded_output=tokenizer.decode(new_output,skip_special_tokens=True)

    return decoded_output


if __name__=='__main__':

    import json
    from message_creator_ginger import message_creator_ginger


    with open('/home/irlab/sagnik/API_KEY','r') as f:
        hf_token_key=f.read()


    ##########################################
    with open('/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/prompt_system_instructions.txt','r') as f:
        out=f.readlines()

    instructions=''

    for i in out:
        instructions+=i




    with open("/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Data/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl","r") as f:
        out=[]
        for i in f:
            out.append(json.loads(i))


    query=out[2]['query']['text']
    passage=out[2]['candidates'][0]['doc']['segment']

    message=message_creator_ginger(instructions,query,passage)
    #############################################

    model='Qwen/Qwen2.5-14B-Instruct'

    output=nugget_generator_ginger(message,model,hf_token_key,passage)

    print(output)
    
    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/samples/sample_nugget_tagged_output.txt','w') as f:
        f.write(output)
