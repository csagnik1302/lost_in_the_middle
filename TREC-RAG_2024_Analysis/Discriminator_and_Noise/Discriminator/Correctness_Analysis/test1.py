import json
import re

out=[]

with open(r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Noise/Correctness_Analysis/misc/pipeline_output_mistral-7b-instruct-v0.3-bnb-4bit.jsonl','r') as f:
    for i in f:
        out.append(json.loads(i))

# out={"model": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "first_gold_doc_pos": 13, "corpus_position": 77, "query": "why did desoto fail", "generator_output": "The DeSoto brand failed for several reasons. One of the main reasons was the 1958 recession, which affected demand for mid-priced automobiles. DeSoto sales were particularly affected, and they failed to recover in 1959 and 1960. Another factor was the competition from other Chrysler brands, such as Plymouth, which offered similar models at a lower price. Additionally, the DeSoto brand had a weak dealer network, as many dealers chose to sell higher-volume Plymouth models instead of slower-selling DeSoto models. Finally, the brand management at Chrysler may have contributed to the failure of DeSoto, as they pitted each of the five Chrysler brands against one another, rather than carefully managing the market to specific price points for all consumers. This lack of focus may have led to confusion among consumers and a lack of clear brand identity for DeSoto.", "nuggetizellm_output": ["1958 recession hurt DeSoto sales\", \"DeSoto sales failed to recover in 1959 and 1960\", \"DeSoto models similar to concurrent Chryslers\", \"Rumors of DeSoto discontinuation\", \"Weakened DeSoto dealer network\", \"Chrysler spun Plymouth off into standalone dealerships\", \"Dealers chose higher-volume Plymouth over DeSoto\", \"DeSoto failed to adjust to changing market trends\", \"No new compact car model in 1960\", \"Chrysler's brand management pitted divisions against each other\", \"Lack of careful market management\", \"General Motors had successful market planning\", \"DeSoto's failure hastened its demise\", \"Compounded by 1961 Newport model introduction\", \"Newport model was an upper-tier DeSoto competitor\", \"DeSoto brand pushed to the brink in 1961"], "nuggetizescorellm_output": ["vital"], "nuggetizeassignerllm_output": ["support"], "scores": {"all_score": 1.0, "all_strict_score": 1.0, "vital_score": 1.0, "vital_strict_score": 1.0, "weighted_score": 1.0, "weighted_strict_score": 1.0}}

# import json
# import re

# INPUT_PATH = r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm.json'
# OUTPUT_PATH = r'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/sample_output_nuggetizellm_fixed.json'


def split_glued_nuggets(raw):
    """
    Handles the case where NuggetizeLLM_output is a list containing ONE
    string with all nuggets joined by '", "' (the bug from the buggy
    parser). Splits it back into individual nugget strings.
    """
    # Strip one layer of leading/trailing quote if present
    raw = raw.strip()
    if raw.startswith('"'):
        raw = raw[1:]
    if raw.endswith('"'):
        raw = raw[:-1]

    # Split on the boundary between nuggets: a quote, comma, space, quote
    parts = re.split(r'"\s*,\s*"', raw)

    # Clean whitespace/newlines inside each part
    parts = [p.replace('\n', ' ').strip() for p in parts]
    parts = [p for p in parts if p]

    return parts


# Possible key names this field has shown up under across different files
NUGGET_KEY_CANDIDATES = [
    'NuggetizeLLM_output',
    'nuggetizellm_output',
]


def find_nugget_key(nugget_dict):
    for key in NUGGET_KEY_CANDIDATES:
        if key in nugget_dict:
            return key
    raise KeyError(
        f"None of the expected keys {NUGGET_KEY_CANDIDATES} found in dict. "
        f"Available keys: {list(nugget_dict.keys())}"
    )


def get_nugget_list(nugget_dict):
    key = find_nugget_key(nugget_dict)
    raw_output = nugget_dict[key]

    # Already a proper list of separate nuggets -> nothing to fix
    if isinstance(raw_output, list) and len(raw_output) > 1:
        return raw_output, key

    # List with a single glued-together string -> needs splitting
    if isinstance(raw_output, list) and len(raw_output) == 1:
        return split_glued_nuggets(raw_output[0]), key

    # Raw string (not even wrapped in a list) -> needs splitting
    if isinstance(raw_output, str):
        return split_glued_nuggets(raw_output), key

    raise ValueError(f"Unexpected type for {key}: {type(raw_output)}")

import os

if __name__ == '__main__':
    # with open(INPUT_PATH, 'r') as f:
    #     nugget_dict = json.load(f)

    for i in out:

        nugget_list, key = get_nugget_list(i)
    
        if len(nugget_list)!=len(i['nuggetizeassignerllm_output']):
            # gen={"first_gold_doc_pos": i['first_gold_doc_pos'], "corpus_position": i['corpus_position']
            print('Itr Not Fine')
        else:
            print('ITr FIne')