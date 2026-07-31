# import matplotlib.pyplot as plt

# with open("/home/irlab/sagnik/Factoid_Analysis/Project/Plots/generated_output_qa_{doc_count}.txt","r") as f:
#     out=f.readlines()


# out=[int(i[:-3]) for i in out]


# ## Plot

# positions = [1, 5, 10, 15, 20, 25, {doc_count}]

# accuracies = [100 * x / 2655 for x in out]

# plt.figure(figsize=(8, 5))

# plt.plot(
#     positions,
#     accuracies,
#     marker="o"
# )

# plt.xticks(positions)

# plt.xlabel("Gold QA Position")
# plt.ylabel("Accuracy (%)")
# plt.title("Llama 3.1 8B QA Retrieval Accuracy vs Position")

# plt.ylim(0, 100)

# plt.grid(True)

# plt.savefig("/home/irlab/sagnik/Factoid_Analysis/Project/Plots/qa_retrieval_accuracy_{doc_count}_docs.png", dpi={doc_count}0, bbox_inches="tight")

# plt.show()


import json
import warnings
import logging as py_logging

from transformers import pipeline
from transformers import logging as hf_logging

import matplotlib.pyplot as plt

from prompt_creation_qa import prompt_qa
from response_matching import best_subspan_em


warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
py_logging.getLogger("httpx").setLevel(py_logging.ERROR)
py_logging.getLogger("huggingface_hub").setLevel(py_logging.ERROR)
py_logging.basicConfig(level=py_logging.ERROR)

############################################### IMPORTS ############################################### 

doc_count=30

PATHS=[rf'/home/irlab/sagnik/Factoid_Analysis/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_0.jsonl',
rf'/home/irlab/sagnik/Factoid_Analysis/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_4.jsonl',
rf'/home/irlab/sagnik/Factoid_Analysis/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_9.jsonl',
rf'/home/irlab/sagnik/Factoid_Analysis/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_14.jsonl',
rf'/home/irlab/sagnik/Factoid_Analysis/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_19.jsonl',
rf'/home/irlab/sagnik/Factoid_Analysis/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_24.jsonl',
rf'/home/irlab/sagnik/Factoid_Analysis/Project/QA/Data/{doc_count}/nq-open-{doc_count}_total_documents_gold_at_29.jsonl',]

with open(PATHS[0],"r") as f:
    out=[]
    for i in f:
        out.append(json.loads(i))


with open(r"/home/irlab/sagnik/API_KEY",'r') as f:
    TOKEN=f.read()

######################################## MODEL IMPLEMENTATION ###################################################

model="meta-llama/Meta-Llama-3.1-8B-Instruct"
generator=pipeline("text-generation",model=model,token=TOKEN)


out1=0
counter=0

for j in PATHS:

    for i in range(len(out)):
        
        prompt,answers=prompt_qa(j,i)

        inputs=generator.tokenizer(prompt, return_tensors='pt')
        context_length=inputs["input_ids"].shape[1]

        out1+=context_length
        counte+=1


print(out1/context_length)

