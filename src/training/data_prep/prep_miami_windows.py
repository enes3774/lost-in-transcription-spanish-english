#!/usr/bin/env python3
"""Build dev-like long-form training windows from the Bangor Miami corpus.

The prepared Miami manifest is utterance-level (median 2.3 s). The competition dev
clips are 5-35 s of continuous single-speaker voice note, so training directly on
2 s utterances teaches a model to stop early -- exactly the truncation failure seen
in the public `whisper-small-spanglish` checkpoint. This script concatenates
consecutive same-speaker utterances into windows of up to `--max-seconds`.

Text is cleaned of the residual CHAT artefacts (IPA `&` fragments, `xxx`,
underscore-joined names, parenthesised elided letters) and otherwise kept verbatim:
Miami casing and punctuation already match what the official scorer expects.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

AMP_TOKEN = re.compile(r"(?<!\w)&\S*")
MULTISPACE = re.compile(r"\s+")


def clean_text(text: str) -> str | None:
    if "xxx" in text or "www" in text:
        return None
    text = AMP_TOKEN.sub(" ", text)
    text = text.replace("_", " ")
    text = text.replace("(", "").replace(")", "")
    text = MULTISPACE.sub(" ", text).strip()
    return text or None


def build_windows(rows: list[dict], max_seconds: float, max_gap: float) -> list[list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["conv"], row["spk"])].append(row)

    windows: list[list[dict]] = []
    for key in sorted(grouped):
        utterances = sorted(grouped[key], key=lambda r: r["start"])
        current: list[dict] = []
        total = 0.0
        for utt in utterances:
            gap = utt["start"] - current[-1]["end"] if current else 0.0
            too_long = total + utt["duration"] > max_seconds
            too_far = gap > max_gap
            if current and (too_long or too_far):
                windows.append(current)
                current, total = [], 0.0
            current.append(utt)
            total += utt["duration"]
        if current:
            windows.append(current)
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--miami-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=24.0)
    parser.add_argument("--max-gap", type=float, default=5.0)
    parser.add_argument("--join-silence", type=float, default=0.12)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.miami_dir / "manifest.jsonl").open(encoding="utf-8")]
    folds = json.loads((args.miami_dir / "split_spk3.json").read_text(encoding="utf-8"))["conv_fold"]

    kept = []
    for row in rows:
        text = clean_text(row["text"])
        if text:
            kept.append({**row, "text": text})
    print(f"utterances: {len(rows)} -> {len(kept)} after cleaning")

    windows = build_windows(kept, args.max_seconds, args.max_gap)
    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(16_000 * args.join_silence), dtype=np.float32)

    manifests = {"train": [], "val": []}
    stats = {"train": 0.0, "val": 0.0}
    for window in windows:
        conv = window[0]["conv"]
        # Fold T is the held-out speaker-disjoint Miami validation set.
        split = "val" if folds.get(conv) == "T" else "train"
        pieces = []
        for i, utt in enumerate(window):
            audio, sample_rate = sf.read(args.miami_dir / "clips" / f"{utt['id']}.wav", dtype="float32")
            assert sample_rate == 16_000, sample_rate
            if i:
                pieces.append(silence)
            pieces.append(audio)
        audio = np.concatenate(pieces)
        duration = len(audio) / 16_000
        if duration < args.min_seconds:
            continue
        name = f"{window[0]['id']}__n{len(window)}.wav"
        sf.write(clips_dir / name, audio, 16_000, subtype="PCM_16")
        manifests[split].append(
            {
                "audio_filepath": str(clips_dir / name),
                "text": " ".join(u["text"] for u in window),
                "duration": duration,
                "conv": conv,
                "spk": window[0]["spk"],
                "n_utts": len(window),
            }
        )
        stats[split] += duration

    for split, items in manifests.items():
        path = args.out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        durations = [i["duration"] for i in items]
        print(
            f"{split}: {len(items)} windows, {stats[split]/3600:.2f} h, "
            f"median {np.median(durations):.1f}s, mean {np.mean(durations):.1f}s -> {path}"
        )


if __name__ == "__main__":
    main()
