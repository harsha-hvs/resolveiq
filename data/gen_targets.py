"""
Step 2: Generate Micro-Postmortem gold targets using Llama-3.3-70B on Together AI
Run:  TOGETHER_API_KEY=xxx python data/gen_targets.py --split train
      TOGETHER_API_KEY=xxx python data/gen_targets.py --split val
      TOGETHER_API_KEY=xxx python data/gen_targets.py --split test
Cost: ~$3.50 total for 1,500 examples
"""
import json, re, time, os, argparse
from pathlib import Path

TEACHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# ── Exact 4-header format the proposal specifies ──────────────────────────
REQUIRED_HEADERS = [
    "## Issue Summary",
    "## Root Cause",
    "## Resolution Steps",
    "## Prevention and Action Items",
]

SYSTEM_PROMPT = """You are a technical writer generating enterprise IT knowledge base articles.

Given an IT incident resolution thread, write a structured Micro-Postmortem with EXACTLY these four sections in this order:

## Issue Summary
## Root Cause
## Resolution Steps
## Prevention and Action Items

Rules:
- Begin DIRECTLY with "## Issue Summary" — no preamble, no greeting, no "Certainly!"
- Every claim must be grounded in the provided thread — do not hallucinate tools, commands, or values
- Maximum 400 tokens total output
- Resolution Steps must be numbered (1. 2. 3. ...)
- Be concise and precise — no filler phrases"""

def make_client():
    """Return a Together AI client. Raise clearly if key is missing."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set TOGETHER_API_KEY environment variable before running.\n"
            "  export TOGETHER_API_KEY=your_key_here"
        )
    from together import Together
    return Together(api_key=api_key)

def passes_filter(text: str) -> bool:
    """
    Quality gate from the proposal:
    (1) All 4 headers present in the correct order
    (2) Token count <= 450  (uses word count if tokenizer unavailable)
    (3) No chatbot preamble (first non-blank line must be ## Issue Summary)
    """
    # Check headers in order
    positions = [text.find(h) for h in REQUIRED_HEADERS]
    if any(p == -1 for p in positions):
        return False
    if positions != sorted(positions):
        return False

    # Check preamble (class12 DPO insight: filler is a trained failure mode)
    first_line = text.strip().split("\n")[0].strip()
    if first_line != "## Issue Summary":
        return False

    # Token length check (approx via words; Llama tokenizer ≈ 1.3 words/token)
    word_count = len(text.split())
    if word_count > 450 * 0.75:  # conservative word-based proxy for 450 tokens
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            if len(tok.encode(text)) > 450:
                return False
        except Exception:
            if word_count > 340:  # fallback: 340 words ≈ 450 tokens
                return False

    return True

def generate_target(client, thread_input: str, temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": thread_input},
        ],
        max_tokens=512,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default="train", choices=["train", "val", "test"])
    parser.add_argument("--data",    default="data/")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    input_path  = Path(args.data) / f"{args.split}.jsonl"
    output_path = Path(args.data) / f"{args.split}_labeled.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Run build_dataset.py first — {input_path} not found")

    with open(input_path) as f:
        examples = [json.loads(line) for line in f]
    print(f"Generating targets for {len(examples)} {args.split} examples...")

    # Resume from checkpoint if partial output exists
    done_inputs = set()
    out_examples = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                ex = json.loads(line)
                out_examples.append(ex)
                done_inputs.add(ex["input"][:100])
        print(f"Resuming — {len(out_examples)} already done")

    client = make_client()
    skipped = 0

    with open(output_path, "a") as out_f:
        for i, ex in enumerate(examples):
            if ex["input"][:100] in done_inputs:
                continue

            success = False
            for attempt in range(args.retries):
                temp = 0.3 + attempt * 0.25   # increase temp on retries
                try:
                    target = generate_target(client, ex["input"], temperature=temp)
                    if passes_filter(target):
                        record = {"input": ex["input"], "target": target}
                        out_f.write(json.dumps(record) + "\n")
                        out_f.flush()
                        out_examples.append(record)
                        success = True
                        break
                    else:
                        print(f"  [{i}] attempt {attempt+1} failed filter (temp={temp:.2f})")
                except Exception as e:
                    print(f"  [{i}] API error: {e} — retrying in 2s")
                    time.sleep(2)

            if not success:
                skipped += 1
                print(f"  [{i}] SKIPPED after {args.retries} attempts")

            if i % 50 == 0 and i > 0:
                print(f"  Progress: {i}/{len(examples)} | kept: {len(out_examples)} | skipped: {skipped}")

            time.sleep(0.12)   # ~500 rpm rate limit buffer

    kept = len(out_examples)
    total = len(examples)
    print(f"\n✓ {args.split} complete: {kept}/{total} passed filter ({100*kept/total:.1f}%)")
    print(f"  Skipped: {skipped} | Output: {output_path}")

    # Warn if too few examples survived
    if args.split == "train" and kept < 800:
        print("\n⚠️  WARNING: fewer than 800 training examples survived.")
        print("   Per proposal decision gate: reduce to 800-example SFT, skip DPO.")

if __name__ == "__main__":
    main()
