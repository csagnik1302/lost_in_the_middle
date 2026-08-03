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

PATHS=[r'/home/irlab/sagnik/lost_in_the_middle/Project/QA/Data/30/nq-open-30_total_documents_gold_at_0.jsonl',
r'/home/irlab/sagnik/lost_in_the_middle/Project/QA/Data/30/nq-open-30_total_documents_gold_at_4.jsonl',
r'/home/irlab/sagnik/lost_in_the_middle/Project/QA/Data/30/nq-open-30_total_documents_gold_at_9.jsonl',
r'/home/irlab/sagnik/lost_in_the_middle/Project/QA/Data/30/nq-open-30_total_documents_gold_at_14.jsonl',
r'/home/irlab/sagnik/lost_in_the_middle/Project/QA/Data/30/nq-open-30_total_documents_gold_at_19.jsonl',
r'/home/irlab/sagnik/lost_in_the_middle/Project/QA/Data/30/nq-open-30_total_documents_gold_at_24.jsonl',
r'/home/irlab/sagnik/lost_in_the_middle/Project/QA/Data/30/nq-open-30_total_documents_gold_at_29.jsonl',]

with open(PATHS[0],"r") as f:
    out=[]
    for i in f:
        out.append(json.loads(i))


with open(r"/home/irlab/sagnik/API_KEY",'r') as f:
    TOKEN=f.read()

######################################## MODEL IMPLEMENTATION ###################################################

model="meta-llama/Meta-Llama-3.1-8B-Instruct"
generator=pipeline("text-generation",model=model,token=TOKEN)


correct_array=[]
PATH_INDEX=1

for j in PATHS:

    correct_pred_count=0

    for i in range(len(out)):
        
        prompt , answers=prompt_qa(j,i)

        response = generator(prompt, return_full_text=False, max_new_tokens=70, do_sample=False)[0]['generated_text']

        best_subspan_score=best_subspan_em(prediction=response, ground_truths=answers)

        correct_pred_count += best_subspan_score

        print(f"DONE: Iteration : {i} for PATH INDEX : {PATH_INDEX} (Response: {best_subspan_score})")

    correct_array.append(correct_pred_count)
    PATH_INDEX+=1


with open("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/generated_output_qa_30.txt", "w") as f:
    for value in correct_array:
        f.write(str(value) + "\n")

################# PLOT #########################################

positions = [1, 5, 10, 15, 20, 25, 30]

accuracies = [100 * x / len(out) for x in correct_array]

plt.figure(figsize=(8, 5))

plt.plot(
    positions,
    accuracies,
    marker="o"
)

plt.xticks(positions)

plt.xlabel("Gold QA Position")
plt.ylabel("Accuracy (%)")
plt.title("Llama 3.1 8B QA Retrieval Accuracy vs Position")

plt.ylim(0, 100)

plt.grid(True)

plt.savefig("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/qa_retrieval_accuracy_30_docs.png", dpi=300, bbox_inches="tight")

plt.show()
