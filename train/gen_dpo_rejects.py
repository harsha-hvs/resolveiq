"""
Step 4a: Generate verbose rejection pairs from the SFT checkpoint
Run:  python train/gen_dpo_rejects.py

Class12 taught us: DPO needs (prompt, chosen, rejected) triples.
The rejected output should come from the SAME model family as chosen
(post-SFT, not base) to prevent the DPO signal conflating format with brevity.
"""
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SFT_CKPT  = Path("checkpoints/sft_v1")
DATA_DIR  = Path("data/")
N_PAIRS   = 500

# System prompt that deliberately induces the failure mode we want to penalize
# Class12 insight: DPO "rejected" outputs should target the *specific* bad behavior
VERBOSE_SYSTEM = """You are a verbose technical writer. Write a very long, over-explained response.
Include conversational filler like "Great question!", "As you can see from the above thread,",
"It's worth noting that", "In conclusion,". Elaborate extensively on every point.
Add extra context not in the thread. Use passive voice. Repeat key information.
Begin with a friendly preamble before getting to the content."""

def load_sft_model():
    print(f"Loading SFT checkpoint from {SFT_CKPT}...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_CKPT)
    tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        revision="8c22764",
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(SFT_CKPT))
    model.eval()
    return model, tokenizer

def generate_verbose(model, tokenizer, thread_input, max_new_tokens=700):
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{VERBOSE_SYSTEM}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{thread_input}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1700).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)

def main():
    out_path = DATA_DIR / "dpo_pairs.jsonl"

    # Load chosen (gold targets from gen_targets.py)
    train_labeled = [json.loads(l) for l in open(DATA_DIR / "train_labeled.jsonl")]
    examples = train_labeled[:N_PAIRS]
    print(f"Generating {N_PAIRS} verbose rejection pairs...")

    model, tokenizer = load_sft_model()
    pairs = []

    with open(out_path, "w") as f:
        for i, ex in enumerate(examples):
            rejected = generate_verbose(model, tokenizer, ex["input"])
            pair = {
                "prompt":   ex["input"],
                "chosen":   ex["target"],    # concise gold target
                "rejected": rejected,         # verbose SFT output
            }
            f.write(json.dumps(pair) + "\n")
            f.flush()
            pairs.append(pair)

            if i % 50 == 0:
                chosen_len   = len(ex["target"].split())
                rejected_len = len(rejected.split())
                print(f"  [{i}/{N_PAIRS}] chosen={chosen_len} words, rejected={rejected_len} words")

    print(f"\n✓ {len(pairs)} DPO pairs written to {out_path}")
    avg_ratio = sum(len(p["rejected"].split()) / max(len(p["chosen"].split()), 1)
                    for p in pairs) / len(pairs)
    print(f"  Average rejected/chosen length ratio: {avg_ratio:.2f}x")
    if avg_ratio < 1.3:
        print("  ⚠️  Rejected outputs are not much longer than chosen.")
        print("     Consider increasing verbose system prompt aggressiveness.")

if __name__ == "__main__":
    main()
