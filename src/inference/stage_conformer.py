"""Conformer-CTC (Miami fine-tuned) stage, plain PyTorch port + SentencePiece.
usage: stage_conformer.py <model_dir> <data_dir> <out_csv>
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, load_audio, log, read_manifest, write_csv  # noqa: E402

import sentencepiece as spm  # noqa: E402
import torch  # noqa: E402
from conformer_port import load_conformer  # noqa: E402

model_dir, data_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = read_manifest(data_dir)
order = sorted(rows, key=lambda r: -float(r.get("file_duration_seconds") or 0))
t0 = time.time()
m = load_conformer(str(model_dir / "conformer_miami.pt"))
sp = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
pool = ThreadPoolExecutor(max_workers=8)
preds = {}
B = 8
for i in range(0, len(order), B):
    part = order[i:i + B]
    ws = list(pool.map(lambda r: load_audio(data_dir / "clips" / r["audio_filename"]), part))
    L = torch.tensor([len(w) for w in ws])
    X = torch.zeros(len(ws), int(L.max()))
    for k, w in enumerate(ws):
        X[k, : len(w)] = torch.from_numpy(w)
    lp, el = m(X.cuda(), L.cuda())
    for r, ids in zip(part, m.greedy_ids(lp, el)):
        preds[r["audio_filename"]] = sp.decode(ids).strip()
write_csv(out_csv, rows, preds)
log(f"conformer stage done in {time.time() - t0:.0f}s")
