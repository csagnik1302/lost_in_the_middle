from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# Only lm-format-enforcer's framework-independent core is imported. Its `integrations.transformers` module is deliberately
# NOT used: it imports names that transformers 5 removed and then reports a misleading "transformers is not installed".
# The small transformers glue it provides is re-implemented below (_build_tokenizer_data / _PrefixFn).
from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData
from nuggetizellm_prompt_creator import prompt_creator_nuggetizellm   # must be the version asking for {"nuggets": [...]}
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator
from typing import Annotated, List
from functools import lru_cache
import json
import re
import torch

#####################

# ---------------------------------------------------------------------------
# Config (rules mirror the prompt: 1-12 words, no redundancy, at most 30 nuggets)
# ---------------------------------------------------------------------------
MIN_WORDS = 1
MAX_WORDS = 12
MIN_NUGGETS = 1
MAX_NUGGETS = 30
MAX_CHARS = 120            # hard cap per nugget, also written into the JSON schema so the grammar bounds output length
SIM_THRESHOLD = 0.8        # word-set Jaccard similarity at/above which two nuggets count as redundant
MAX_NEW_TOKENS = 1400      # 30 nuggets x ~30 tokens + JSON syntax fits comfortably
MAX_ATTEMPTS = 3           # attempt 1 is greedy; retries use sampling
OFFLOAD_CACHE = True       # set False if generate() complains about cache_implementation="offloaded" with constrained decoding


# ---------------------------------------------------------------------------
# Output type: used BOTH as the decoding grammar (via its JSON schema) and as the semantic validator
# ---------------------------------------------------------------------------
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


class NuggetList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Field constraints become minItems / maxItems / minLength / maxLength in the schema the decoder enforces
    nuggets: List[Annotated[str, StringConstraints(min_length=1, max_length=MAX_CHARS)]] = Field(
        ..., min_length=MIN_NUGGETS, max_length=MAX_NUGGETS
    )

    # Semantic rules the grammar cannot express. Order: normalise -> drop junk -> word range -> drop (near-)duplicates -> cap
    @field_validator("nuggets", mode="before")
    @classmethod
    def enforce_rules(cls, v):
        if not isinstance(v, list):
            raise ValueError("nuggets must be a list")

        kept, kept_sets = [], []

        for item in v:
            if not isinstance(item, str):
                continue

            n = _BULLET.sub("", " ".join(item.split()))     # collapse whitespace, drop "- " / "1. " prefixes

            if not re.search(r"\w", n):                     # no letters/digits -> junk
                continue
            if len(n) > MAX_CHARS or not (MIN_WORDS <= len(n.split()) <= MAX_WORDS):
                continue

            ws = set(re.findall(r"\w+", n.lower()))
            if any(len(ws & k) / len(ws | k) >= SIM_THRESHOLD for k in kept_sets):
                continue                                    # redundant with an earlier (more important) nugget

            kept.append(n)
            kept_sets.append(ws)

            if len(kept) == MAX_NUGGETS:                    # keep the most important (earliest) ones
                break

        if not kept:
            raise ValueError("no valid nuggets left after enforcing rules")
        return kept


SCHEMA = NuggetList.model_json_schema()


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
_TOK_DATA = {}   # tokenizer-data build is slow, so do it once per tokenizer


def _build_tokenizer_data(tokenizer):
    vocab_size = len(tokenizer)
    token_0 = tokenizer.encode("0")[-1]
    special = set(tokenizer.all_special_ids)
    regular_tokens = []
    for idx in range(vocab_size):
        if idx in special:
            continue
        # prepend token "0" and drop its first character to see whether this token starts a new word (leading space)
        after_0 = tokenizer.decode([token_0, idx])[1:]
        regular = tokenizer.decode([idx])
        regular_tokens.append((idx, after_0, len(after_0) > len(regular)))
    decode_fn = lambda toks: tokenizer.decode(toks).rstrip('\ufffd')
    try:
        return TokenEnforcerTokenizerData(regular_tokens, decode_fn, tokenizer.eos_token_id, False, vocab_size)
    except TypeError:   # older lm-format-enforcer signature without use_bitmask / vocab_size
        return TokenEnforcerTokenizerData(regular_tokens, decode_fn, tokenizer.eos_token_id)


def _tokenizer_data(tokenizer):
    key = id(tokenizer)
    if key not in _TOK_DATA:
        _TOK_DATA[key] = _build_tokenizer_data(tokenizer)
    return _TOK_DATA[key]


# prefix_allowed_tokens_fn for generate(): receives (batch_id, full token sequence) and returns the allowed next token ids
class _PrefixFn:
    def __init__(self, enforcer):
        self.enforcer = enforcer

    def __call__(self, batch_id, sent):
        allowed = self.enforcer.get_allowed_tokens(sent.tolist())
        return getattr(allowed, "allowed_tokens", allowed)


# Stops generation as soon as the output is a complete, parseable JSON value, so the model can't ramble or loop afterwards
class JsonDoneCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.tok = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        done = False
        last = self.tok.convert_ids_to_tokens(int(input_ids[0, -1]))
        if last and '}' in last:                            # only pay for a decode when a closing brace was just produced
            text = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
            try:
                json.loads(text)
                done = True
            except json.JSONDecodeError:
                pass
        return torch.tensor([done], device=input_ids.device)


def _generate(model, tokenizer, inputs, input_size, sample):
    # a fresh parser + prefix fn per generate() call (the enforcer keeps per-generation state)
    parser = JsonSchemaParser(SCHEMA)
    prefix_fn = _PrefixFn(TokenEnforcer(_tokenizer_data(tokenizer), parser))

    kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        prefix_allowed_tokens_fn=prefix_fn,
        stopping_criteria=StoppingCriteriaList([JsonDoneCriteria(tokenizer, input_size)]),
        pad_token_id=tokenizer.eos_token_id,
    )
    if OFFLOAD_CACHE:
        kwargs["cache_implementation"] = "offloaded"
    if sample:
        kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
    else:
        kwargs.update(do_sample=False)

    output_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **kwargs)
    return tokenizer.decode(output_ids[0][input_size:], skip_special_tokens=True)


def _validate(output):
    """Returns (nuggets or None, error message or None, number of raw items the model produced or None)."""
    try:
        raw_n = len(json.loads(output)["nuggets"])
    except Exception:
        raw_n = None
    try:
        return NuggetList.model_validate_json(output).nuggets, None, raw_n
    except ValidationError as e:
        return None, str(e), raw_n


@lru_cache(maxsize=None)
def load_retr_set(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Main entry point (same signature and output keys as before, plus a 'meta' dict)
# ---------------------------------------------------------------------------
def NuggetizeLLM(corpus_lookup_index, model, tokenizer, retr_set_path):

    input_data = load_retr_set(retr_set_path)[corpus_lookup_index]

    prompt, query = prompt_creator_nuggetizellm(input_data)
    query = query.replace('\n', '')

    input_processed = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    # add_special_tokens=False: the chat template already contains <s>, so the default would add a second BOS
    inputs = tokenizer(input_processed, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_size = inputs['input_ids'].shape[1]

    nuggets, err, raw_n = None, None, None
    attempts = 0

    for attempt in range(MAX_ATTEMPTS):
        attempts = attempt + 1
        output = _generate(model, tokenizer, inputs, input_size, sample=(attempt > 0))
        nuggets, err, raw_n = _validate(output)
        if nuggets is not None:
            break
        print(f"[index {corpus_lookup_index}] attempt {attempts} failed validation: {err}")

    dropped = (raw_n - len(nuggets)) if (nuggets is not None and raw_n is not None) else None

    nugget_dict = {
        'query': query,
        'NuggetizeLLM_output': nuggets,      # list[str], or None if every attempt failed
        'meta': {'attempts': attempts, 'dropped_by_rules': dropped, 'error': err},
    }

    return nugget_dict




if __name__=="__main__":

    from dotenv import load_dotenv
    import os

    load_dotenv()
    hf_token=os.getenv('HF_TOKEN')

    retr_set_path=r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Data/bm25/generator_input_data_gold_fixed_3_app2A.jsonl'

    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    model=AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,attn_implementation='flash_attention_2')
    tokenizer=AutoTokenizer.from_pretrained(model_name,token=hf_token)

    nugget_dict=NuggetizeLLM(0,model,tokenizer,retr_set_path)

    print(nugget_dict)

    with open(r'/home/irlab/sagnik/Non_Factoid_Analysis/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Correctness_Analysis/misc/bm25/sample_output_nuggetizellm.json','w') as f:
        json.dump(nugget_dict,f,indent=2)