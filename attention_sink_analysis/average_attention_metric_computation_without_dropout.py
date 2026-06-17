import sys

sys.path.append("/home/irlab/sagnik/lost_in_the_middle/Project/QA")

from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_creation_qa import prompt_qa
import matplotlib.pyplot as plt
import torch


with open(r"/home/irlab/sagnik/API_KEY","r") as f:
    TOKEN_KEY=f.read()



prompt_count=1
gold_count=0
doc_count=10

PATH=f"lost_in_the_middle/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_{gold_count}.jsonl"


prompts=[]
for i in range(prompt_count):
    prompts.append(prompt_qa(PATH,i)[0])

def measure_attention_sink(model,prompts,tokenizer,device=torch.device("cuda")):
    
    num_layers=model.config.num_hidden_layers     # Number of layers in the model
    num_heads=model.config.num_attention_heads    # Number of attention heads per layer

    inputs=[]

    for i in tqdm(prompts):
        input=tokenizer(i,return_tensors="pt").to(device)      # return_tensors="pt" returns the output in pytorch tensor form
        inputs.append(input)

    outputs=[]

    for i in tqdm(inputs):
        output=model.generate(**i,output_attentions=True,return_dict_in_generate=True,max_new_tokens=1)     # **i: Unpacks the kv data stored in dictionary i and makes it ready to use as a input, return_dict_in_generate returns the output in dictionary form which is better and much structured way of outputting stuff when we are outputting stuff other than the just output tokens
        outputs.append(output)

    return [i.attentions for i in outputs],num_heads,num_layers,[i.input_ids.size() for i in inputs]






model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

model=AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="eager",token=TOKEN_KEY).to(torch.device("cuda"))
tokenizer=AutoTokenizer.from_pretrained(model_name,token=TOKEN_KEY)

output,num_heads,num_layers,token_counts1 = measure_attention_sink(model=model,prompts=prompts,tokenizer=tokenizer)


############
token_counts=[]

for i in range(prompt_count):
    token_counts.append(token_counts1[i][1])

min_token_count=min(token_counts)
###############


epsilon=0.3

sink_indicator_prompt=torch.zeros(min_token_count).to(torch.device("cuda"))


for l in tqdm(range(prompt_count)):
    sink_indicator_layer=torch.zeros(min_token_count).to(torch.device("cuda"))

    for k in tqdm(range(num_layers)):
        sink_indicator_head=torch.zeros(min_token_count).to(torch.device("cuda"))
        tensor=output[l][0][k]

        for i in range(num_heads):
            matrix=tensor[0][i]
            importance_score=[]

            for j in range(min_token_count):
                temp=matrix[j:,j]
                mean=torch.mean(temp).item()
                importance_score.append(mean)
            
            importance_score_tensor=torch.tensor(importance_score).to(torch.device("cuda"))
            sink_indicator_token=(importance_score_tensor>epsilon).float()

            sink_indicator_head+=sink_indicator_token
            
        sink_indicator_average_head=(1/num_heads)*sink_indicator_head

        sink_indicator_layer+=sink_indicator_average_head

    sink_indicator_average_layer=(1/num_layers)*sink_indicator_layer

    sink_indicator_prompt+=sink_indicator_average_layer

    torch.save((1/(l+1))*sink_indicator_prompt,f"/home/irlab/sagnik/attention_sink_analysis/Plot/Without_Dropout/{model_name}/sink_indicator_average_layer_prompt_count_{prompt_count}_doc_count_{doc_count}_gold_{gold_count}.pt")

sink_indicator_average_prompt=(1/prompt_count)*sink_indicator_prompt

torch.save(sink_indicator_average_prompt,f"/home/irlab/sagnik/attention_sink_analysis/Plot/Without_Dropout/{model_name}/sink_indicator_average_layer_prompt_count_{prompt_count}_doc_count_{doc_count}_gold_{gold_count}.pt")

### Plot

xcount=sink_indicator_average_prompt.size()[0]

xaxis=list(range(xcount+1))[1:]

plt.figure(figsize=(8, 5))

plt.plot(
    xaxis,
    sink_indicator_average_prompt.to(torch.device("cpu")),
    marker="o"
)

plt.xticks(xaxis)

plt.xlabel("Prompt Token Position")
plt.ylabel("Average Attention Score (Across All Prompts)")
plt.title("Attention Score")

plt.grid(True)

plt.savefig(f"/home/irlab/sagnik/attention_sink_analysis/Plot/Without_Dropout/{model_name}/attention_score_prompt_count_{prompt_count}_doc_count_{doc_count}_gold_{gold_count}.png", dpi=300, bbox_inches="tight")
