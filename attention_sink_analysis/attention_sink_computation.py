from transformers import AutoTokenizer, AutoModelForCausalLM


def measure_attention_sink(model):
    
    num_layers=model.config.num_hidden_layers

    return num_layers


model=AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

print(measure_attention_sink(model))
