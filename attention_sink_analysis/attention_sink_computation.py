import sys

sys.path.append("/home/irlab/sagnik/lost_in_the_middle/Project/QA")

from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_creation_qa import prompt_qa
import torch


PATH="lost_in_the_middle/Project/QA/Data/10/nq-open-10_total_documents_gold_at_0.jsonl"

prompts=[]
for i in range(3):
    prompts.append(prompt_qa(PATH,1)[0])


def measure_attention_sink(model,prompts,tokenizer,device=torch.device("cuda")):
    
    num_layers=model.config.num_hidden_layers     # Number of layers in the model
    num_heads=model.config.num_attention_heads    # Number of attention heads per layer

    inputs=[]

    for i in tqdm(prompts):
        input=tokenizer(i,return_tensors="pt").to(device)      # return_tensors="pt" returns the output in pytorch tensor form
        inputs.append(input)

    outputs=[]

    for i in tqdm(inputs):
        output=model.generate(**i,output_attentions=True,return_dict_in_generate=True).to(device)     # **i: Unpacks the kv data stored in dictionary i and makes it ready to use as a input, return_dict_in_generate returns the output in dictionary form which is better and much structured way of outputting stuff when we are outputting stuff other than the just output tokens
        outputs.append(output)

    return outputs

model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"

model=AutoModelForCausalLM.from_pretrained(model_name).to(torch.device("cuda"))
tokenizer=AutoTokenizer.from_pretrained(model_name).to(torch.device("cuda"))


print(measure_attention_sink(model=model,prompts=prompts,tokenizer=tokenizer)[0])


