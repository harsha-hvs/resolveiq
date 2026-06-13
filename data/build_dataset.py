"""
Step 1: Build train/val/test splits from Server Fault Posts.xml
Run:  python data/build_dataset.py --xml data/Posts.xml --out data/
"""
import xml.etree.ElementTree as ET
import json, re, random, argparse
from pathlib import Path

# ── Try to import tokenizer for length-capping; fall back to word count ──
try:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    def count_tokens(text): return len(_tok.encode(text))
    def truncate(text, max_tok=1500): return _tok.decode(_tok.encode(text)[:max_tok])
    print("Using Llama tokenizer for length control")
except Exception:
    def count_tokens(text): return len(text.split())
    def truncate(text, max_tok=1500): return " ".join(text.split()[:max_tok])
    print("Tokenizer unavailable — using word count fallback")

MAX_INPUT_TOKENS = 1500

def clean_html(text):
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r'<code>(.*?)</code>', lambda m: m.group(1), text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_posts(xml_path):
    """Parse Posts.xml into a dict keyed by post Id."""
    print(f"Parsing {xml_path} ...")
    posts = {}
    tree = ET.iterparse(xml_path, events=("start",))
    for i, (event, elem) in enumerate(tree):
        if elem.tag != "row":
            continue
        a = elem.attrib
        pid = a.get("Id")
        if not pid:
            continue
        posts[pid] = {
            "type":     a.get("PostTypeId", ""),
            "body":     clean_html(a.get("Body", "")),
            "parent":   a.get("ParentId", ""),
            "score":    int(a.get("Score", "0")),
            "accepted": a.get("AcceptedAnswerId", ""),
            "title":    clean_html(a.get("Title", "")),
        }
        elem.clear()
        if i % 100000 == 0:
            print(f"  parsed {i:,} rows, {len(posts):,} posts so far")
    print(f"Done. Total posts: {len(posts):,}")
    return posts

def build_threads(posts):
    """
    For each question with an accepted answer, build a flattened thread.
    Format mirrors the proposal spec: [QUESTION] ... [ANSWER_1] ... etc.
    """
    questions = {k: v for k, v in posts.items()
                 if v["type"] == "1" and v["accepted"] and len(v["body"]) > 100}
    print(f"Questions with accepted answers: {len(questions):,}")

    threads = []
    for qid, q in questions.items():
        # Accepted answer first, then top-voted answers (up to 3 total)
        accepted = posts.get(q["accepted"])
        if not accepted or len(accepted["body"]) < 50:
            continue

        other_answers = sorted(
            [v for v in posts.values()
             if v["type"] == "2" and v["parent"] == qid
             and v["score"] >= 1 and len(v["body"]) > 50],
            key=lambda x: -x["score"]
        )[:2]

        thread = f"[QUESTION] {q['body']}\n"
        thread += f"[ANSWER_1] {accepted['body']}\n"
        for i, ans in enumerate(other_answers, 2):
            thread += f"[ANSWER_{i}] {ans['body']}\n"

        thread = truncate(thread.strip(), MAX_INPUT_TOKENS)

        # Quality gate: skip very short threads
        if count_tokens(thread) < 100:
            continue

        threads.append({"input": thread})

    print(f"Valid threads built: {len(threads):,}")
    return threads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",  default="data/Posts.xml")
    parser.add_argument("--out",  default="data/")
    parser.add_argument("--n",    type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    posts   = parse_posts(args.xml)
    threads = build_threads(posts)

    random.seed(args.seed)
    random.shuffle(threads)
    threads = threads[:args.n]

    n_train = int(args.n * 0.80)   # 1200
    n_val   = int(args.n * 0.10)   # 150
    # rest   = test                 # 150

    splits = {
        "train": threads[:n_train],
        "val":   threads[n_train:n_train + n_val],
        "test":  threads[n_train + n_val:],
    }

    for split, data in splits.items():
        path = Path(args.out) / f"{split}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(data):>5} examples → {path}")

    print("\n✓ Dataset build complete.")
    print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

if __name__ == "__main__":
    main()
