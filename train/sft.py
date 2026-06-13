import json, torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    BitsAndBytesConfig, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
REVISION  = "8c22764"
DATA_DIR  = Path("data/")
CKPT_DIR  = Path("checkpoints/sft_v1")

TARGET_MODS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

def load_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f]

def format_prompt(ex):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are ResolveIQ, an enterprise IT knowledge base writer. "
        "Given an incident resolution thread, write a structured Micro-Postmortem "
        "with exactly four sections: Issue Summary, Root Cause, Resolution Steps, "
        "and Prevention and Action Items. Be concise and grounded in the thread."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{ex['input']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{ex['target']}<|eot_id|>"
    )

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading 4-bit Llama-3.1-8B-Instruct (commit 8c22764)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=TARGET_MODS,
        lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # LoRA sanity check
    print("\n[sanity] Verifying LoRA at init produces zero delta...")
    inputs = tokenizer("Test", return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out_lora = model(**inputs).logits
        with model.disable_adapter():
            out_base = model(**inputs).logits
    delta = (out_lora - out_base).abs().max().item()
    print(f"[sanity] max |LoRA(x) - base(x)| at init = {delta:.2e}  (expect ~0)")
    model.train()

    train_data = load_jsonl(DATA_DIR / "train_labeled.jsonl")
    val_data   = load_jsonl(DATA_DIR / "val_labeled.jsonl")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_ds = Dataset.from_list([{"text": format_prompt(ex)} for ex in train_data])
    val_ds   = Dataset.from_list([{"text": format_prompt(ex)} for ex in val_data])

    args = SFTConfig(
        output_dir=str(CKPT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="resolveiq-sft-v1",
        max_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n=== Starting SFT training ===")
    trainer.train()
    trainer.save_model(str(CKPT_DIR))
    tokenizer.save_pretrained(str(CKPT_DIR))
    print(f"\n✓ SFT complete → {CKPT_DIR}")

if __name__ == "__main__":
    main()
