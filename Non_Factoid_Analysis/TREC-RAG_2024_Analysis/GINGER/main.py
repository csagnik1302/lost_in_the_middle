import json
from message_creator_ginger import message_creator_ginger
from nugget_generator_ginger import passage_generator_ginger,nugget_extractor_ginger
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

###### Dataset Import 

with open("/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Data/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl","r") as f:
    data=[]
    for i in f:
        data.append(json.loads(i))


##### PROMPTS

with open('/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/schema_and_related_inputs/prompt_system_instructions.txt','r') as f:
    out=f.readlines()

prompt1=''

for i in out:
    prompt1+=i


prompt_set=[]
passage_set=[]

for i in range(5):

    query=data[0]['query']['text']
    
    passage=data[0]['candidates'][i]['doc']['segment']
    passage_set.append(passage)

    out=message_creator_ginger(prompt1,query,passage)
    prompt_set.append(out)



######## Nugget_Passage_Generation

with open('/home/irlab/sagnik/API_KEY','r') as f:
    hf_token_key=f.read()

model_name='Qwen/Qwen2.5-14B-Instruct'

model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,token=hf_token_key)
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name,token=hf_token_key)

annotated_passages=[]

count=1
for i,j in zip(prompt_set,passage_set):
    annotated_passage=passage_generator_ginger(i,model,tokenizer,j)
    annotated_passages.append(annotated_passage)
    print(f'Passage {count} Annotation DONE')
    count+=1

    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Pipeline_outputs/annotated_passages','w') as f:
        for i in annotated_passages:
            f.write(i)



########### Nugget_Generation

extracted_nuggets=[]

nugget_dict={}

for i in annotated_passages:
    nugget=nugget_extractor_ginger(i)
    if len(nugget)>0:
        extracted_nuggets.extend(nugget)
        nugget_dict[query]=extracted_nuggets
    
    with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/GINGER/Pipeline_outputs/nuggets.jsonl','w') as f:
        for i,j in nugget_dict.items():
            temp={}
            temp[i]=j
            json.dump(temp,f)
            f.write('\n')