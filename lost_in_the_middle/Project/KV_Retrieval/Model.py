import json
import re
import warnings
import logging as py_logging

from transformers import pipeline
from transformers import logging as hf_logging

import matplotlib.pyplot as plt

from prompt_creation_kv import prompt


warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
py_logging.getLogger("httpx").setLevel(py_logging.ERROR)
py_logging.getLogger("huggingface_hub").setLevel(py_logging.ERROR)
py_logging.basicConfig(level=py_logging.ERROR)

############################################### IMPORTS ############################################### 

with open(r"/home/irlab/sagnik/lost_in_the_middle/Project/KV_Retrieval/Data/kv-retrieval-300_keys.jsonl","r") as f:
  data=[]
  for i in f:
    data.append(json.loads(i))


with open(r"/home/irlab/sagnik/API_KEY",'r') as f:
    TOKEN=f.read()


######################################## MODEL IMPLEMENTATION ###################################################
stop=False

model="meta-llama/Meta-Llama-3.1-8B-Instruct"


generator=pipeline("text-generation",model=model,token=TOKEN)


correct_array=[]
positions = [0, 49, 99, 149, 199, 249, 299]

for gold in positions:

   correct_count=0


   for i in range(500):

      PROMPT, gold_value = prompt(data, i, gold)

      out = generator(
         PROMPT,
         return_full_text=False,
         max_new_tokens=70,
         do_sample=False
      )

      generated_output = out[0]["generated_text"]

      match = re.search(
         r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
         generated_output
      )


      if match is None:
         print(f"No UUID found: {generated_output}")
         continue

      generated_output = match.group(0)

      if generated_output == gold_value:
         correct_count += 1

      print(
         f"DONE: Iteration: {i} for gold_position: {gold}"
      )

   correct_array.append(correct_count)

   with open("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/generated_output_kv_300.txt", "w") as f:
      for value in correct_array:
         f.write(str(value) + "\n")

################# PLOT #########################################

positions = [1, 50, 100, 150, 200, 250, 300]

accuracies = [100 * x / 500 for x in correct_array]

plt.figure(figsize=(8, 5))

plt.plot(
    positions,
    accuracies,
    marker="o"
)

plt.xticks(positions)

plt.xlabel("Gold KV Position")
plt.ylabel("Accuracy (%)")
plt.title("Llama 3.1 8B KV Retrieval Accuracy vs Position")

plt.ylim(0, 100)

plt.grid(True)

plt.savefig("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/kv_retrieval_accuracy_300.png", dpi=300, bbox_inches="tight")

plt.show()
