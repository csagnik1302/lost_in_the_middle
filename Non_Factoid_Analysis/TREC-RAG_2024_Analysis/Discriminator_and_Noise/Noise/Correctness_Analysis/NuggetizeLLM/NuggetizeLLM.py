from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)

from nuggetizellm_prompt_creator import prompt_creator_nuggetizellm

from functools import lru_cache

import ast
import json
import re
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 1400

OFFLOAD_CACHE = True

# Minimum number of nuggets we expect from the parser.
# This is NOT used to delete nuggets.
MIN_NUGGETS = 1


# ---------------------------------------------------------------------------
# Robust nugget parser
# ---------------------------------------------------------------------------

def _clean_nugget(text):
    """
    Light normalization only.

    IMPORTANT:
    This does not apply word limits, character limits,
    similarity filtering, or any other nugget-dropping rule.
    """

    if not isinstance(text, str):
        return None

    text = text.strip()

    # Remove common surrounding whitespace.
    text = text.strip()

    if not text:
        return None

    # Remove common bullet/number prefixes.
    text = re.sub(
        r'^\s*(?:[-*•]|\d+[.)])\s+',
        '',
        text
    ).strip()

    if not text:
        return None

    return text


def _extract_quoted_strings(text):
    """
    Character-level extraction of quoted strings.

    Supports:
        "text"
        'text'

    Handles:
        commas inside strings
        escaped quotes
        escaped backslashes

    Example:

        '"nugget 1","nugget 2","nugget, with comma"'

    becomes:

        [
            'nugget 1',
            'nugget 2',
            'nugget, with comma'
        ]
    """

    nuggets = []

    i = 0
    n = len(text)

    while i < n:

        # Look for either quote type.
        if text[i] not in ('"', "'"):
            i += 1
            continue

        quote = text[i]
        i += 1

        chars = []

        while i < n:

            ch = text[i]

            # Closing quote
            if ch == quote:
                i += 1
                break

            # Escape sequence
            if ch == '\\' and i + 1 < n:

                next_ch = text[i + 1]

                # Preserve common escaped characters correctly.
                escape_map = {
                    'n': '\n',
                    'r': '\r',
                    't': '\t',
                    '\\': '\\',
                    '"': '"',
                    "'": "'"
                }

                if next_ch in escape_map:
                    chars.append(escape_map[next_ch])
                else:
                    # Unknown escape:
                    # retain the escaped character rather than dropping it.
                    chars.append(next_ch)

                i += 2
                continue

            chars.append(ch)
            i += 1

        value = ''.join(chars)

        value = _clean_nugget(value)

        if value:
            nuggets.append(value)

    return nuggets


def _extract_nugget_array(text):
    """
    Try to isolate the contents of:

        "nuggets": [...]

    even when the surrounding JSON is malformed.
    """

    # Locate the nuggets field.
    match = re.search(
        r'["\']?nuggets["\']?\s*:',
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    start = match.end()

    # Find the first '[' after "nuggets":
    array_start = text.find('[', start)

    if array_start == -1:
        return None

    # Find matching closing bracket while respecting quotes.
    depth = 0
    quote = None
    escaped = False

    for i in range(array_start, len(text)):

        ch = text[i]

        if quote is not None:

            if escaped:
                escaped = False
                continue

            if ch == '\\':
                escaped = True
                continue

            if ch == quote:
                quote = None

            continue

        if ch in ('"', "'"):
            quote = ch
            continue

        if ch == '[':
            depth += 1

        elif ch == ']':
            depth -= 1

            if depth == 0:
                return text[
                    array_start + 1:i
                ]

    # Unterminated array:
    # return everything after '[' rather than losing the nuggets.
    return text[array_start + 1:]


def _parse_with_json(text):
    """
    First-choice parser for proper JSON.
    """

    try:

        obj = json.loads(text)

    except Exception:

        return None

    if isinstance(obj, dict):

        nuggets = obj.get("nuggets")

        if isinstance(nuggets, list):

            return [
                x for x in nuggets
                if isinstance(x, str)
            ]

    elif isinstance(obj, list):

        return [
            x for x in obj
            if isinstance(x, str)
        ]

    return None


def _parse_with_literal_eval(text):
    """
    Handles Python-style structures, including single quotes.

    Example:

        {'nuggets': ['A', 'B']}

    """

    try:

        obj = ast.literal_eval(text)

    except Exception:

        return None

    if isinstance(obj, dict):

        nuggets = obj.get("nuggets")

        if isinstance(nuggets, list):

            return [
                x for x in nuggets
                if isinstance(x, str)
            ]

    elif isinstance(obj, list):

        return [
            x for x in obj
            if isinstance(x, str)
        ]

    # Important case:
    #
    # '"nugget 1","nugget 2"'
    #
    # literal_eval returns the inner sequence as a string.
    if isinstance(obj, str):

        return _extract_quoted_strings(obj)

    return None


def _parse_nuggets(output):
    """
    Main tolerant nugget parser.

    The parser deliberately tries increasingly permissive
    strategies instead of rejecting the entire output.

    Returns:
        list[str]
    """

    if output is None:
        return []

    if not isinstance(output, str):
        output = str(output)

    text = output.strip()

    if not text:
        return []

    # ---------------------------------------------------------------
    # 1. Proper JSON
    # ---------------------------------------------------------------

    nuggets = _parse_with_json(text)

    if nuggets is not None:

        cleaned = [
            _clean_nugget(x)
            for x in nuggets
        ]

        return [
            x for x in cleaned
            if x is not None
        ]

    # ---------------------------------------------------------------
    # 2. Python-style representation
    #
    # Handles single quotes.
    # ---------------------------------------------------------------

    nuggets = _parse_with_literal_eval(text)

    if nuggets is not None:

        cleaned = [
            _clean_nugget(x)
            for x in nuggets
        ]

        return [
            x for x in cleaned
            if x is not None
        ]

    # ---------------------------------------------------------------
    # 3. Recover the "nuggets" array from malformed JSON
    # ---------------------------------------------------------------

    array_content = _extract_nugget_array(text)

    if array_content is not None:

        nuggets = _extract_quoted_strings(
            array_content
        )

        if nuggets:
            return nuggets

    # ---------------------------------------------------------------
    # 4. Direct quoted-string sequence
    #
    # Handles:
    #
    # "A","B","C"
    #
    # and:
    #
    # '"A","B","C"'
    # ---------------------------------------------------------------

    nuggets = _extract_quoted_strings(text)

    if nuggets:

        # Avoid treating a single outer wrapper containing
        # the entire sequence as the only nugget.
        #
        # Example:
        #
        # '"A","B","C"'
        #
        # The scanner gives:
        #
        # ['A","B","C']
        #
        # In this situation, recursively parse the content.
        if len(nuggets) == 1:

            inner = nuggets[0]

            inner_nuggets = _extract_quoted_strings(
                inner
            )

            if len(inner_nuggets) > 1:
                return inner_nuggets

        return nuggets

    # ---------------------------------------------------------------
    # 5. Last-resort line-based recovery
    #
    # We do NOT discard the output simply because it wasn't
    # valid JSON.
    # ---------------------------------------------------------------

    lines = text.splitlines()

    recovered = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        cleaned = _clean_nugget(line)

        if cleaned:
            recovered.append(cleaned)

    return recovered


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class JsonDoneCriteria(StoppingCriteria):

    def __init__(self, tokenizer, prompt_len):

        self.tok = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):

        # We no longer require valid JSON to stop.
        #
        # Stop when the model reaches EOS.
        #
        # Returning False here allows normal max_new_tokens
        # generation to proceed.

        return torch.tensor(
            [False],
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

    kwargs = dict(

        max_new_tokens=MAX_NEW_TOKENS,

        # -----------------------------------------------------------
        # GREEDY DECODING
        # -----------------------------------------------------------

        do_sample=False,

        pad_token_id=tokenizer.eos_token_id
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
# Main nugget generation
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

    inputs = tokenizer(
        input_processed,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    input_size = inputs["input_ids"].shape[1]

    # ---------------------------------------------------------------
    # Generate ONCE, greedily.
    # ---------------------------------------------------------------

    raw_output = _generate(
        model,
        tokenizer,
        inputs,
        input_size
    )

    # ---------------------------------------------------------------
    # Recover nuggets without Pydantic.
    # ---------------------------------------------------------------

    nuggets = _parse_nuggets(
        raw_output
    )

    nugget_dict = {

        "query": query,

        "NuggetizeLLM_output": nuggets,

        "meta": {

            "raw_output": raw_output,

            "raw_nuggets_recovered": len(nuggets),

            "parser": "tolerant_multi_stage",

            "error": None
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

    print("\n================ RAW OUTPUT ================\n")

    print(
        nugget_dict["meta"]["raw_output"]
    )

    print("\n================ PARSED NUGGETS ================\n")

    for i, nugget in enumerate(
        nugget_dict["NuggetizeLLM_output"],
        start=1
    ):

        print(
            f"{i}. {nugget}"
        )

    print(
        "\nRecovered nuggets:",
        len(
            nugget_dict["NuggetizeLLM_output"]
        )
    )

    output_path = (
        r"/home/irlab/sagnik/"
        r"Non_Factoid_Analysis/"
        r"TREC-RAG_2024_Analysis/"
        r"Discriminator_and_Noise/"
        r"Noise/"
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
            indent=2,
            ensure_ascii=False
        )