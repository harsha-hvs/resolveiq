"""
Step 4b: Direct Preference Optimization to penalize verbosity
Run:  python train/dpo.py

Class12 DPO loss (Rafailov et al. 2023):
L_DPO(θ) = -E[ log σ( β * log(π_θ(y_w|x)/π_ref(y_w|x))
                      - β * log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]

where y_w = chosen (concise), y_l = rejected (verbose).
This directly maximizes the margin between the log-prob of concise vs verbose outputs.
The Z(x) partition function cancels — no explicit reward model needed.
"""
import torch
from pathlib import Path
from datasets import Dataset
import json

SFT_CKPT = Path("checkpoints/sft_v1")
DPO_CKPT = Path("checkpoints/dpo_v1")
DATA_DIR  = Path("data/")

# DPO hypers
BETA          = 0.1     # KL penalty weight; lower = more aggressive update
EPOCHS        = 2
BATCH_SIZE    = 2
GRAD_ACCUM    = 8       # effective batch = 16 (same as SFT)
LR            = 5e-5    # lower than SFT; we're fine-tuning a fine-tune
MAX_PROMPT_LEN = 1600
MAX_TARGET_LEN = 512

def main():
    DPO_CKPT.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_CKPT)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO trainer requires left-padding

    # ── Load SFT model as the starting point + reference model ───────────
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        revision="8c22764",
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Load the SFT adapters onto the base
    model = PeftModel.from_pretrained(base, str(SFT_CKPT), is_trainable=True)
    model.enable_input_require_grads()

    # DPO with PEFT: pass ref_model=None, TRL uses the base (frozen) as reference.
    # This avoids loading a second full model copy.

    # ── Load DPO pairs ────────────────────────────────────────────────────
    pairs = [json.loads(l) for l in open(DATA_DIR / "dpo_pairs.jsonl")]
    print(f"Loaded {len(pairs)} DPO preference pairs")

    # TRL DPOTrainer expects keys: prompt, chosen, rejected
    dataset = Dataset.from_list(pairs)

    # ── DPO config ────────────────────────────────────────────────────────
    from trl import DPOTrainer, DPOConfig

    dpo_config = DPOConfig(
        output_dir=str(DPO_CKPT),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        beta=BETA,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        run_name="resolveiq-dpo-v1",
        
        
        # loss_type="sigmoid" is the standard DPO loss from class12 §G.4
        loss_type="sigmoid",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # use the frozen base via PEFT's implicit reference
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n=== Starting DPO training ===")
    print(f"β = {BETA} | Epochs = {EPOCHS} | ~4 min on H200")
    print("Objective: maximize margin between concise (chosen) and verbose (rejected)")

    trainer.train()
    trainer.save_model(str(DPO_CKPT))
    tokenizer.save_pretrained(str(DPO_CKPT))
    print(f"\n✓ DPO complete. Checkpoint saved to {DPO_CKPT}")

if __name__ == "__main__":
    main()
