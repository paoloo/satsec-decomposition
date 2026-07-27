"""LoRA SFT for the grounded decomposition model.

Fine-tunes an instruct base (default Qwen2.5-1.5B-Instruct) on the decomposition
tuning set produced by build_tuning_set.py. The model learns to turn an objective
and retrieved grounding into an ordered, standards-traceable plan. Completion masking
excludes prompt tokens from the token-level loss; it does not prevent prompt facts from
influencing gradients or guarantee that facts are absent from the learned parameters.

Designed to run inside the paolo-dev container on atadev:
    docker exec satsec-train sh -lc \
      "cd /workspace/satsec && HF_HOME=/workspace/satsec/hf \
       python3 train_lora.py --data tuning_set.jsonl --out models/satsec-decomp-1.5b"

Tiny data (a few dozen examples) means this first run is a PIPELINE-VALIDATION pilot,
not a final model. Train on the 'train' split only; the 'test' split is held out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_split(path: str, split: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        if ex.get("meta", {}).get("split", "train") == split:
            rows.append(ex)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tuning_set.v2.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--model-revision", required=True,
                    help="immutable Hugging Face commit for the base model")
    ap.add_argument("--out", default="models/satsec-decomp-1.5b")
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lr-scheduler", default="linear")
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument(
        "--save-strategy", choices=("no", "epoch"), default="epoch",
        help="use 'no' for short LOCO folds; the final adapter is still saved",
    )
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    print(f"[train] base={args.base_model} cuda={cuda} epochs={args.epochs}")

    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_rows = read_split(args.data, "train")
    if not train_rows:
        raise SystemExit(f"no train-split rows in {args.data}")
    print(f"[train] {len(train_rows)} training examples")

    def to_features(ex):
        """Completion-masked: loss only on the assistant turn, prompt masked to -100."""
        msgs = ex["messages"]
        prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_ids = tok(prompt_text, truncation=True, max_length=args.max_seq_len)["input_ids"]
        full_ids = tok(full_text, truncation=True, max_length=args.max_seq_len)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[: len(full_ids)]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    features = [to_features(ex) for ex in train_rows]
    if any(not any(label != -100 for label in ex["labels"]) for ex in features):
        raise SystemExit("at least one example lost its entire completion to truncation")
    ds_tok = Dataset.from_list(features)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # needed for grad checkpointing + LoRA

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=QWEN_LORA_TARGETS, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim=args.optim,
        logging_steps=5,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=cuda,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_tok,
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
    )
    trainer.train()
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    data_sha256 = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    manifest = {
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "dataset": str(Path(args.data).resolve()),
        "dataset_sha256": data_sha256,
        "train_examples": len(train_rows),
        "quantization": None,
        "precision": "bfloat16" if cuda else "float32",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "config": vars(args),
        "target_modules": QWEN_LORA_TARGETS,
        "completion_only_labels": True,
        "completion_mask_caveat": (
            "Prompt positions are excluded from token-level loss, but prompt activations "
            "still condition completion losses; this is not a no-memorization guarantee."
        ),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    print(f"[train] saved LoRA adapter -> {out}")


if __name__ == "__main__":
    main()
