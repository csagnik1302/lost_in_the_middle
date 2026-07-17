# # # import json

# # # with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','r') as f:
# # #     out=json.load(f)

# # # for i in range(3):
# # #     for j in range(2):
# # #         out['i']+=1
# # #         out['j']+=1

# # # with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/itr_index.json','w') as f:
# # #     json.dump(out,f)

# # # print(out)


# # try:
# #     l={"hi":"Hello"My name""}
# # except Exception as e:
# #     print('Error')

# import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import json
# import gzip
# import torch
# import re
# import os
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings(
#     "ignore",
#     category=FutureWarning,
#     module="bitsandbytes"
# )
# import sys

# input=[]

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_error_log_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         input.append(json.loads(i))

# indices_missed=[]

# for i in input:
#     indices_missed.append([i['first_gold_doc_pos'],i['corpus_position']])

# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/Generator')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/NuggetizeLLM')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/NuggetizeScoreLLM')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/NuggetizeAssignerLLM')
# sys.path.insert(0,r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/Evaluation')

# from generator import llm_output_generator
# from NuggetizeLLM import NuggetizeLLM
# from NuggetizeScoreLLM import NuggetizeScoreLLM
# from NuggetizeAssignerLLM import NuggetizeAssignerLLM
# from evaluator import all_score, all_strict_score, vital_score, vital_strict_score, weighted_score, weighted_strict_score

# #####################
# with open(r'/home/irlab/sagnik/API_KEY','r') as f:
#     hf_token=f.read()

# retr_set_path=r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Data/generator_input_data_gold_fixed_3.jsonl'

# model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
# model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
# tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

# length_set=[]

# with open(retr_set_path,'r') as f:
#     for i in f:
#         temp=json.loads(i)
#         length_set.append(temp)  


# # ########################

# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/itr_index.json','r') as f:
#     itr=json.load(f)

# # test_indices=[0,6,13,19,25,32,38,44,51,57]
# # # test_indices=[0]

# for i,j in indices_missed:

#     # if i<itr['i']:
#     #     continue

#     # if i==itr['i'] and j<=itr['j']:
#     #     continue
#     output_text, query=llm_output_generator(i,j,model,tokenizer,retr_set_path)
#     output1=output_text

#     nugget_dict=NuggetizeLLM(j,model,tokenizer,retr_set_path)
#     output2=nugget_dict['NuggetizeLLM_output']

#     score_list, nugget_list, query1=NuggetizeScoreLLM(model,tokenizer,nugget_dict)
#     output3=score_list

#     assigner_list, nugget_list1, query2=NuggetizeAssignerLLM(model,tokenizer,nugget_dict,output_text)
#     output4=assigner_list


#     ######## Eval

#     all_score_out=all_score(assigner_list)
#     all_strict_score_out=all_strict_score(assigner_list)

#     vital_score_out=vital_score(assigner_list,score_list)
#     vital_strict_score_out=vital_strict_score(assigner_list,score_list)

#     weighted_score_out=weighted_score(assigner_list,score_list)
#     weighted_strict_score_out=weighted_strict_score(assigner_list,score_list)

#     output5={'all_score':all_score_out,'all_strict_score':all_strict_score_out,'vital_score':vital_score_out,'vital_strict_score':vital_strict_score_out,'weighted_score':weighted_score_out,'weighted_strict_score':weighted_strict_score_out}

#     final_output={'model':model_name,'first_gold_doc_pos':i,'corpus_position':j,'query':query,'generator_output':output1,'nuggetizellm_output':output2,'nuggetizescorellm_output':output3,'nuggetizeassignerllm_output':output4,'scores':output5}


#     with open(fr"/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_{model_name[model_name.index('/')+1:]}.jsonl", "a") as f:
#         f.write(json.dumps(final_output) + "\n")
#         f.flush()
#         os.fsync(f.fileno())


#################################
################################

# import json

# inp1=[]
# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         inp1.append(json.loads(i))

# inp2=[]
# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         inp2.append(json.loads(i))

# for i in range(len(inp1)):
#     a1=inp1[i]['first_gold_doc_pos']
#     a2=inp1[i]['corpus_position']
#     a3=inp1[i]['generator_output']

#     for j in range(len(inp2)):
#         b1=inp2[j]['first_gold_doc_pos']
#         b2=inp2[j]['corpus_position']
#         b3=inp2[j]['generator_output']

#         if a1==b1 and a2==b2:
#             if a3==b3:
#                 print(f'Duplicate Queries: {a1},{a2} | {b1},{b2}')
#                 break

# print('ALL DONE')

######################################33
##########################################

# import json

# inp1=[]
# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         inp1.append(json.loads(i))

# inp2=[]
# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         inp2.append(json.loads(i))


# l1=[]

# for i in inp2:
#     temp1=i['first_gold_doc_pos']
#     temp2=i['corpus_position']

#     l1.append([temp1,temp2])

# l2=[]
# for i in l1:
#     l2.append(l1.count(i))

# print(set(l2))

################################################
##################################################


# import json

# inp1=[]
# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         inp1.append(json.loads(i))

# inp2=[]
# with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
#     for i in f:
#         inp2.append(json.loads(i))


# k=[]

# itr=inp1

# for i in itr:
#     temp1=i['first_gold_doc_pos']
#     if temp1 not in k:
#         k.append(temp1)


# out=[]
# for i in k:
#     count=0
#     for j in itr:
#         temp2=j['first_gold_doc_pos']
        
#         if temp2==i:
#             count+=1
    
#     out.append(count)


# print(k,out)

#####################################################
#####################################################


import json

inp1=[]
with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
    for i in f:
        inp1.append(json.loads(i))

inp2=[]
with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
    for i in f:
        inp2.append(json.loads(i))

k=[]

itr=inp1

out=[]

for j in itr:
    temp2=j['first_gold_doc_pos']
    temp3=j['corpus_position']

    if temp2==25:
        if temp3 not in out:
            out.append(temp3)


# ref=list(range(125))

# for i in out:
#     if i not in ref:
#         print(i)

print(sorted(out),len(sorted(out)))
