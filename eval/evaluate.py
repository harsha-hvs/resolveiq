"""
Enhanced evaluation:
- Class 13 enhancement: proper stop-string handling to prevent over-generation
- Class 16 enhancement: quantitative faithfulness score (every command in
  Resolution Steps must trace back to the input thread)
- Standard metrics from §5.2: format adherence, BERTScore F1, verbosity penalty

Run:  python eval/evaluate.py
"""
import json, re, torch, argparse
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList,
)
from peft import PeftModel

REQUIRED_HEADERS = [
    "## Issue Summary", "## Root Cause",
    "## Resolution Steps", "## Prevention and Action Items",
]

MODEL_CONFIGS = {
    "baseline":   {"path": "meta-llama/Llama-3.1-8B-Instruct", "revision": "8c22764", "peft": None},
    "post_sft":   {"path": "checkpoints/sft_v1",                "revision": None,      "peft": "checkpoints/sft_v1"},
    "dpo_merged": {"path": "checkpoints/merged_final",          "revision": None,      "peft": None},
}

# ── Class 13 enhancement: stop on EOT or second "## Issue Summary" ────────
class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)

# ── Metric functions ──────────────────────────────────────────────────────

def check_format(text):
    positions = [text.find(h) for h in REQUIRED_HEADERS]
    if any(p == -1 for p in positions): return 0
    return int(positions == sorted(positions))

def check_preamble(text):
    """Output must begin directly with ## Issue Summary."""
    first = text.strip().split("\n")[0].strip()
    return int(first == "## Issue Summary")

# Class 16 enhancement: quantitative faithfulness
COMMAND_PATTERNS = [
    r'`([^`\n]{2,50})`',                 # backtick-quoted
    r'\b(sudo\s+\S+)',                   # sudo commands
    r'\b(systemctl\s+\S+\s+\S+)',        # systemctl ...
    r'\b(/etc/[\w/.\-]+)',               # config paths
    r'\b(\w+_\w+\s*=\s*\d+)',            # config assignments
]

def extract_commands(text):
    """Extract command-like tokens from a Resolution Steps block."""
    found = set()
    for pat in COMMAND_PATTERNS:
        for m in re.finditer(pat, text):
            tok = m.group(1).strip()
            if 2 < len(tok) < 80:
                found.add(tok)
    return found

def faithfulness_score(pred, source):
    """
    Class 16 inspired: for every command/path/config in Resolution Steps,
    check the *core token* appears in the input thread.
    Returns (score in [0,1], unsourced_commands list).
    """
    steps_match = re.search(r"## Resolution Steps\s*\n(.*?)(?=##\s|\Z)", pred, re.DOTALL)
    if not steps_match:
        return 1.0, []   # no steps = nothing to verify
    steps = steps_match.group(1)
    commands = extract_commands(steps)
    if not commands:
        return 1.0, []
    source_lower = source.lower()
    unsourced = []
    for cmd in commands:
        # Extract distinctive token (longest alphanumeric chunk)
        tokens = re.findall(r'[a-zA-Z_][\w./\-]{2,}', cmd)
        if not tokens:
            continue
        anchor = max(tokens, key=len)
        if anchor.lower() not in source_lower:
            unsourced.append(cmd)
    score = 1.0 - (len(unsourced) / len(commands))
    return score, unsourced

def verbosity_flag(pred, ref, tokenizer, threshold=0.15):
    pred_len = len(tokenizer.encode(pred))
    ref_len  = len(tokenizer.encode(ref))
    return int(pred_len > ref_len * (1 + threshold))

def run_inference(model, tokenizer, input_text):
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1700).to(model.device)
    # Class 13: stop on a second occurrence of any header (over-generation guard)
    stop = StoppingCriteriaList([
        StopOnStrings(["<|eot_id|>"], tokenizer),
    ])
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    # Post-trim: if the model re-emits "## Issue Summary" after producing one
    # full article, cut at the second occurrence.
    second = text.find("## Issue Summary", text.find("## Issue Summary") + 1)
    if second > 0:
        text = text[:second].strip()
    return text

def load_model(cfg):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_path = cfg["path"]
    tok_path  = cfg["peft"] or base_path

    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(quantization_config=bnb, device_map="auto",
                  torch_dtype=torch.bfloat16)
    if cfg["revision"]:
        kwargs["revision"] = cfg["revision"]
    model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
    if cfg["peft"] and cfg["peft"] != base_path:
        model = PeftModel.from_pretrained(model, cfg["peft"])
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/test_labeled.jsonl")
    parser.add_argument("--out",    default="results/eval.json")
    parser.add_argument("--models", nargs="+",
                        default=["baseline", "post_sft", "dpo_merged"])
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    test_data = [json.loads(l) for l in open(args.data)]
    print(f"Evaluating on {len(test_data)} test examples")

    all_results = {}
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model '{model_name}', skipping"); continue
        print(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")

        cfg = MODEL_CONFIGS[model_name]
        model, tokenizer = load_model(cfg)

        preds, refs, inputs_list = [], [], []
        fmt, preamble, verb, faith_scores = [], [], [], []
        unsourced_log = []

        for i, ex in enumerate(test_data):
            pred = run_inference(model, tokenizer, ex["input"])
            preds.append(pred); refs.append(ex["target"]); inputs_list.append(ex["input"])
            fmt.append(check_format(pred))
            preamble.append(check_preamble(pred))
            verb.append(verbosity_flag(pred, ex["target"], tokenizer))
            fs, unsrc = faithfulness_score(pred, ex["input"])
            faith_scores.append(fs)
            if unsrc: unsourced_log.append({"i": i, "unsourced": unsrc})
            if i % 25 == 0:
                print(f"  [{i}/{len(test_data)}] fmt={fmt[-1]} faith={fs:.2f}")

        # BERTScore on format-passing only
        fp_idx = [i for i, s in enumerate(fmt) if s == 1]
        fp_preds = [preds[i] for i in fp_idx]
        fp_refs  = [refs[i]  for i in fp_idx]
        bert_f1 = 0.0
        if fp_preds:
            from bert_score import score as bscore
            _, _, F1 = bscore(fp_preds, fp_refs,
                             model_type="microsoft/deberta-xlarge-mnli",
                             lang="en", verbose=False)
            bert_f1 = F1.mean().item()

        all_results[model_name] = {
            "format_adherence":     sum(fmt)/len(fmt),
            "bertscore_f1":         bert_f1,
            "verbosity_penalty_rate": sum(verb)/len(verb),
            "preamble_clean_rate":  sum(preamble)/len(preamble),
            "faithfulness_score":   sum(faith_scores)/len(faith_scores),
            "n_format_pass":        len(fp_preds),
            "n_total":              len(test_data),
        }

        # Save predictions + faithfulness log
        with open(f"results/{model_name}_predictions.jsonl", "w") as f:
            for inp, pred, ref, fl, vl, pl, fs in zip(
                inputs_list, preds, refs, fmt, verb, preamble, faith_scores
            ):
                f.write(json.dumps({
                    "input": inp[:300], "prediction": pred, "reference": ref,
                    "format_pass": fl, "verbosity_flag": vl,
                    "preamble_clean": pl, "faithfulness": fs,
                }) + "\n")
        with open(f"results/{model_name}_unsourced.json", "w") as f:
            json.dump(unsourced_log, f, indent=2)

        del model; torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Ablation table — five metrics
    print("\n" + "="*90)
    print("ABLATION TABLE")
    print("="*90)
    hdr = f"{'Model':<22} {'Format':>9} {'BERTScore':>12} {'Verb':>8} {'NoPream':>9} {'Faithful':>10}"
    print(hdr); print("-"*90)
    for name in ["baseline", "post_sft", "dpo_merged"]:
        if name not in all_results: continue
        r = all_results[name]
        print(f"{name:<22} {r['format_adherence']:>9.3f} {r['bertscore_f1']:>12.4f} "
              f"{r['verbosity_penalty_rate']:>8.3f} {r['preamble_clean_rate']:>9.3f} "
              f"{r['faithfulness_score']:>10.3f}")
    print("="*90)

if __name__ == "__main__":
    main()
