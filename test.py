import argparse
import json
import sys

from transformers import AutoConfig, AutoTokenizer

SENTINEL = 10**15  # HF uses ~1e30 for "unset" tokenizer.model_max_length


def text_cfg(cfg):
    """Multimodal models nest the LLM settings under text_config / llm_config."""
    for key in ("text_config", "llm_config", "language_config"):
        sub = getattr(cfg, key, None)
        if sub is not None:
            return sub
    return cfg


def static_check(model_id, trust_remote_code=False):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    t = text_cfg(cfg)

    max_pos = getattr(t, "max_position_embeddings", None)
    rope = getattr(t, "rope_scaling", None) or getattr(t, "rope_parameters", None)
    window = getattr(t, "sliding_window", None)

    effective = max_pos
    notes = []

    if isinstance(rope, dict):
        rtype = rope.get("rope_type") or rope.get("type")
        factor = rope.get("factor")
        orig = rope.get("original_max_position_embeddings")
        notes.append(f"rope_scaling: type={rtype}, factor={factor}, original_max_pos={orig}")
        if factor and orig and max_pos and max_pos < orig * factor:
            # scaling declared but max_pos is the unscaled value
            effective = int(orig * factor)
            notes.append("max_position_embeddings looks unscaled; rope scaling "
                         "in config extends it")
    elif max_pos:
        notes.append("no rope scaling in config (some repos say 'add YaRN to "
                     "extend' in the README - that's opt-in, not default)")

    if window:
        notes.append(f"sliding_window={window} (local attention span; global "
                     "reach may be larger)")

    tok_max = None
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tok_max = tok.model_max_length
        if tok_max and tok_max > SENTINEL:
            tok_max = None  # unset
    except Exception as e:  # tokenizer is optional for this check
        notes.append(f"tokenizer load failed: {e}")

    gen_max = None
    try:
        from transformers import GenerationConfig
        g = GenerationConfig.from_pretrained(model_id)
        gen_max = g.max_length if g.max_length != 20 else None  # 20 = HF default
    except Exception:
        pass

    return {
        "model": model_id,
        "model_type": getattr(cfg, "model_type", None),
        "max_position_embeddings": max_pos,
        "effective_context": effective,
        "tokenizer_model_max_length": tok_max,
        "generation_max_length": gen_max,
        "notes": notes,
    }


def print_report(r):
    print(f"\n=== {r['model']} ({r['model_type']}) ===")
    print(f"  max_position_embeddings : {r['max_position_embeddings']}")
    print(f"  effective context       : {r['effective_context']}")
    print(f"  tokenizer max length    : {r['tokenizer_model_max_length']}")
    print(f"  generation max_length   : {r['generation_max_length']}")
    for n in r["notes"]:
        print(f"  - {n}")


# ---------------------------------------------------------------- needle test
FILLER = ("The quick brown fox jumps over the lazy dog while the committee "
          "reviews the quarterly logistics report. ")


def build_prompt(tok, target_tokens, needle_pos=0.5):
    needle = "The secret passcode is MANGO-7421. "
    per = max(1, len(tok.encode(FILLER, add_special_tokens=False)))
    reps = max(2, (target_tokens - 120) // per)
    cut = int(reps * needle_pos)
    body = FILLER * cut + needle + FILLER * (reps - cut)
    return body + "\n\nWhat is the secret passcode? Answer with the passcode only."


def needle_test(model_id, endpoint, served_name, claimed, fractions, api_key, trust_remote_code):
    import requests
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    print(f"\nNeedle test on {served_name} @ {endpoint}")
    for f in fractions:
        target = int(claimed * f)
        prompt = build_prompt(tok, target)
        n = len(tok.encode(prompt, add_special_tokens=False))
        try:
            resp = requests.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": served_name,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 20, "temperature": 0},
                timeout=900,
            )
            resp.raise_for_status()
            out = resp.json()["choices"][0]["message"]["content"]
            ok = "MANGO-7421" in out
            print(f"  {f:>4.0%} (~{n:>7} tok): {'PASS' if ok else 'FAIL'}  -> {out.strip()[:60]!r}")
        except Exception as e:
            print(f"  {f:>4.0%} (~{n:>7} tok): ERROR {str(e)[:120]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="HF model ids or local paths")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a report")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    ap.add_argument("--served-name", help="model name the server exposes (default: first model id)")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--fractions", default="0.25,0.5,0.75,0.95",
                    help="fractions of the claimed window to test")
    args = ap.parse_args()

    results = []
    for m in args.models:
        try:
            r = static_check(m, args.trust_remote_code)
        except Exception as e:
            print(f"[{m}] failed: {e}", file=sys.stderr)
            continue
        results.append(r)
        if not args.json:
            print_report(r)

    if args.json:
        print(json.dumps(results, indent=2))

    if args.endpoint and results:
        r = results[0]
        claimed = r["effective_context"] or r["tokenizer_model_max_length"]
        if not claimed:
            sys.exit("No claimed context length found; cannot run needle test.")
        fr = [float(x) for x in args.fractions.split(",")]
        needle_test(r["model"], args.endpoint, args.served_name or r["model"],
                    claimed, fr, args.api_key, args.trust_remote_code)


if __name__ == "__main__":
    main()