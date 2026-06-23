from datasets import load_dataset

with open(r'/home/irlab/sagnik/API_KEY','r') as f:
    hf_token_key=f.read()

ds=load_dataset("BeIR/trec-covid","queries", token=hf_token_key)

ds['queries'].to_json('trec_covid_queries.json')