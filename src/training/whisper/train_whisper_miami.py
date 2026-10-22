#!/usr/bin/env python3
"""Fine-tune a Whisper checkpoint on Bangor Miami code-switched windows.

Targets are the verbatim mixed-language Miami transcripts, so the model is taught
to write each word in the language it was spoken in instead of translating the
minority language into the prompted one.

Supports full fine-tuning (small checkpoints) and LoRA (large-v3).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--language", default="es")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--max-label-tokens", type=int, default=256)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class WindowDataset(Dataset):
    def __init__(self, manifest: Path, processor: WhisperProcessor, max_label_tokens: int):
        self.items = [json.loads(line) for line in manifest.open(encoding="utf-8")]
        self.processor = processor
        self.max_label_tokens = max_label_tokens

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        audio, sample_rate = sf.read(item["audio_filepath"], dtype="float32")
        assert sample_rate == 16_000, sample_rate
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        features = self.processor.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="np", padding="max_length"
        ).input_features[0]
        labels = self.processor.tokenizer(item["text"]).input_ids[: self.max_label_tokens]
        return {"input_features": features, "labels": labels}


@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict:
        batch = {
            "input_features": torch.tensor(
                np.stack([f["input_features"] for f in features]), dtype=torch.float32
            )
        }
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = label_batch.input_ids.masked_fill(label_batch.attention_mask.ne(1), -100)
        # The decoder prepends the start-of-transcript token itself.
        if (labels[:, 0] == self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    processor = WhisperProcessor.from_pretrained(
        args.processor, language=args.language, task="transcribe", local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, local_files_only=True
    )
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.config.use_cache = False

    if args.freeze_encoder:
        model.model.encoder.requires_grad_(False)

    if args.lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        if args.gradient_checkpointing:
            # Checkpointed encoder blocks produce no grad path to the LoRA weights
            # unless the inputs themselves require grad.
            model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    train_set = WindowDataset(args.train_manifest, processor, args.max_label_tokens)
    val_set = WindowDataset(args.val_manifest, processor, args.max_label_tokens)
    print(f"train windows {len(train_set)}  val windows {len(val_set)}", flush=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,
        logging_steps=25,
        report_to=[],
        dataloader_num_workers=6,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=Collator(processor),
    )
    trainer.train()

    final = args.out_dir / "final"
    if args.lora:
        merged = model.merge_and_unload()
        merged.generation_config.language = args.language
        merged.generation_config.task = "transcribe"
        merged.save_pretrained(final)
    else:
        trainer.save_model(str(final))
    processor.save_pretrained(final)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
