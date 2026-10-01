from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)

# Only lm-format-enforcer's framework-independent core is imported.
# Its integrations.transformers module is deliberately NOT used:
# it imports names that transformers 5 removed and then reports a
# misleading "transformers is not installed".
#
# The small transformers glue it provides is re-implemented below
# (_build_tokenizer_data / _PrefixFn).

from lmformatenforcer import (
    JsonSchemaParser,
    TokenEnforcer,
    TokenEnforcerTokenizerData
)

from nuggetizellm_prompt_creator import prompt_creator_nuggetizellm

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from typing import List

from functools import lru_cache

import json
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_NUGGETS = 1
MAX_NUGGETS = 30

MAX_NEW_TOKENS = 1400

# Greedy decoding is deterministic, so one attempt is sufficient.
MAX_ATTEMPTS = 1

OFFLOAD_CACHE = True
# Set False if generate() complains about
# cache_implementation="offloaded" with constrained decoding.


# ---------------------------------------------------------------------------
# Output type
#
# Used BOTH as:
#   1. the decoding grammar via its JSON schema
#   2. the semantic validator
#
# There is NO:
#   - character limit
#   - word limit
#   - similarity/redundancy filtering
#
# The only restrictions are:
#   - nuggets must be a list
#   - at least 1 nugget
#   - at most 30 nuggets
#   - each nugget must be a string
# ---------------------------------------------------------------------------

class NuggetList(BaseModel):

    model_config = ConfigDict(extra="forbid")

    nuggets: List[str] = Field(
        ...,
        min_length=MIN_NUGGETS,
        max_length=MAX_NUGGETS
    )


SCHEMA = NuggetList.model_json_schema()


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

_TOK_DATA = {}
# tokenizer-data build is slow, so do it once per tokenizer


def _build_tokenizer_data(tokenizer):

    vocab_size = len(tokenizer)

    token_0 = tokenizer.encode("0")[-1]

    special = set(tokenizer.all_special_ids)

    regular_tokens = []

    for idx in range(vocab_size):

        if idx in special:
            continue

        # Prepend token "0" and drop its first character
        # to see whether this token starts a new word
        # (leading space).

        after_0 = tokenizer.decode([token_0, idx])[1:]

        regular = tokenizer.decode([idx])

        regular_tokens.append(
            (
                idx,
                after_0,
                len(after_0) > len(regular)
            )
        )

    decode_fn = lambda toks: tokenizer.decode(toks).rstrip("\ufffd")

    try:

        return TokenEnforcerTokenizerData(
            regular_tokens,
            decode_fn,
            tokenizer.eos_token_id,
            False,
            vocab_size
        )

    except TypeError:

        # Older lm-format-enforcer signature without
        # use_bitmask / vocab_size

        return TokenEnforcerTokenizerData(
            regular_tokens,
            decode_fn,
            tokenizer.eos_token_id
        )


def _tokenizer_data(tokenizer):

    key = id(tokenizer)

    if key not in _TOK_DATA:
        _TOK_DATA[key] = _build_tokenizer_data(tokenizer)

    return _TOK_DATA[key]


# ---------------------------------------------------------------------------
# prefix_allowed_tokens_fn for generate()
# ---------------------------------------------------------------------------

class _PrefixFn:

    def __init__(self, enforcer):

        self.enforcer = enforcer

    def __call__(self, batch_id, sent):

        allowed = self.enforcer.get_allowed_tokens(
            sent.tolist()
        )

        return getattr(
            allowed,
            "allowed_tokens",
            allowed
        )


# ---------------------------------------------------------------------------
# Stop generation as soon as the output is a complete,
# parseable JSON value.
# ---------------------------------------------------------------------------

class JsonDoneCriteria(StoppingCriteria):

    def __init__(self, tokenizer, prompt_len):

        self.tok = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):

        done = False

        last = self.tok.convert_ids_to_tokens(
            int(input_ids[0, -1])
        )

        if last and "}" in last:

            text = self.tok.decode(
                input_ids[0, self.prompt_len:],
                skip_special_tokens=True
            )

            try:

                json.loads(text)

                done = True

            except json.JSONDecodeError:

                pass

        return torch.tensor(
            [done],
            device=input_ids.device
        )


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

def _generate(
    model,
    tokenizer,
    inputs,
    input_size
):

    # Fresh parser + prefix function for each generation.
    # The enforcer keeps per-generation state.

    parser = JsonSchemaParser(SCHEMA)

    prefix_fn = _PrefixFn(
        TokenEnforcer(
            _tokenizer_data(tokenizer),
            parser
        )
    )

    kwargs = dict(

        max_new_tokens=MAX_NEW_TOKENS,

        prefix_allowed_tokens_fn=prefix_fn,

        stopping_criteria=StoppingCriteriaList(
            [
                JsonDoneCriteria(
                    tokenizer,
                    input_size
                )
            ]
        ),

        pad_token_id=tokenizer.eos_token_id,

        # ---------------------------------------------------------------
        # GREEDY DECODING
        # ---------------------------------------------------------------
        do_sample=False
    )

    if OFFLOAD_CACHE:

        kwargs["cache_implementation"] = "offloaded"

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **kwargs
    )

    return tokenizer.decode(
        output_ids[0][input_size:],
        skip_special_tokens=True
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(output):

    """
    Returns:

        nuggets
        error message
        number of raw items the model produced
    """

    try:

        raw_n = len(
            json.loads(output)["nuggets"]
        )

    except Exception:

        raw_n = None

    try:

        return (
            NuggetList.model_validate_json(output).nuggets,
            None,
            raw_n
        )

    except ValidationError as e:

        return (
            None,
            str(e),
            raw_n
        )


# ---------------------------------------------------------------------------
# Load retrieval set
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_retr_set(path):

    with open(path, "r") as f:

        return [
            json.loads(line)
            for line in f
        ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def NuggetizeLLM(
    corpus_lookup_index,
    model,
    tokenizer,
    retr_set_path
):

    input_data = load_retr_set(
        retr_set_path
    )[corpus_lookup_index]

    prompt, query = prompt_creator_nuggetizellm(
        input_data
    )

    query = query.replace("\n", "")

    input_processed = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    # The chat template already contains <s>,
    # so don't add another BOS token.

    inputs = tokenizer(
        input_processed,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    input_size = inputs["input_ids"].shape[1]

    nuggets = None
    err = None
    raw_n = None

    attempts = 0

    for attempt in range(MAX_ATTEMPTS):

        attempts = attempt + 1

        # Always greedy.
        output = _generate(
            model,
            tokenizer,
            inputs,
            input_size
        )

        nuggets, err, raw_n = _validate(
            output
        )

        if nuggets is not None:

            break

        print(
            f"[index {corpus_lookup_index}] "
            f"attempt {attempts} failed validation: {err}"
        )

    nugget_dict = {

        "query": query,

        "NuggetizeLLM_output": nuggets,

        "meta": {
            "attempts": attempts,
            "raw_nuggets": raw_n,
            "error": err
        }
    }

    return nugget_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    from dotenv import load_dotenv

    import os

    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    retr_set_path = (
        r"/home/irlab/sagnik/"
        r"Non_Factoid_Analysis/"
        r"TREC-RAG_2024_Analysis/"
        r"Discriminator_and_Noise/"
        r"Data/bm25/"
        r"generator_input_data_gold_fixed_3_app2A.jsonl"
    )

    model_name = (
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )

    nugget_dict = NuggetizeLLM(
        0,
        model,
        tokenizer,
        retr_set_path
    )

    print(nugget_dict)

    output_path = (
        r"/home/irlab/sagnik/"
        r"Non_Factoid_Analysis/"
        r"TREC-RAG_2024_Analysis/"
        r"Discriminator_and_Noise/"
        r"Discriminator/"
        r"Correctness_Analysis/"
        r"misc/bm25/"
        r"sample_output_nuggetizellm.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            nugget_dict,
            f,
            indent=2
        )