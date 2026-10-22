"""submission_v9c.zip vs submission_v9b.zip from the central directories: same entry names; uncompressed size and CRC equal for
every entry except the three Phi-4 shards and the Phi-4 delta files (merged / zero-filled / one-tensor delta). exit 0 / 1."""
import os
import sys
import zipfile

T = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track"


def listing(path):
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
    names = [i.filename for i in infos]
    assert len(names) == len(set(names)), f"duplicate names in {path}"
    return {i.filename: (i.file_size, i.CRC, i.compress_type, i.compress_size) for i in infos}


a, b = listing(f"{T}/submission_v9b.zip"), listing(f"{T}/submission_v9c.zip")
only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
diff = sorted(n for n in set(a) & set(b) if a[n][:2] != b[n][:2])
allowed = {"models/phi4/model-00001-of-00003.safetensors", "models/phi4/model-00002-of-00003.safetensors",
           "models/phi4/model-00003-of-00003.safetensors", "models/phi4_delta/delta.safetensors", "models/phi4_delta/SOURCE.txt"}
size, v8, v9b = os.path.getsize(f"{T}/submission_v9c.zip"), os.path.getsize(f"{T}/submission_v8.zip"), os.path.getsize(f"{T}/submission_v9b.zip")
print(f"entries: v9b {len(a)}, v9c {len(b)}; only in v9b: {only_a[:6]}; only in v9c: {only_b[:6]}")
print(f"entries whose content differs: {diff}")
top = {}
for n, (fs, crc, ct, cs) in b.items():
    k = "/".join(n.split("/")[:2]) if n.startswith("models/") else "(code + vendor)"
    top[k] = (top.get(k, (0, 0))[0] + fs, top.get(k, (0, 0))[1] + cs)
print("per model, GB unpacked -> GB in the zip: " + "; ".join(f"{k.replace('models/', '')} {u / 1e9:.2f} -> {c / 1e9:.2f}" for k, (u, c) in sorted(top.items(), key=lambda x: -x[1][0])))
print(f"unpacked total {sum(v[0] for v in b.values()) / 1e9:.2f} GB; methods: {sorted(set(v[2] for v in b.values()))} (8 = deflate, 0 = stored)")
print(f"zip size {size} bytes = {size / 1e9:.3f} GB | v8 zip (ran) {v8 / 1e9:.3f} GB -> difference {(size - v8) / 1e9:+.3f} GB | v9b zip (hung) {v9b / 1e9:.3f} GB")
ok = not only_a and not only_b and set(diff) <= allowed and "main.py" in b
print("ZIPCMP_PASSED: v9c zip == v9b zip except the Phi-4 weight files" if ok else "ZIPCMP_FAILED")
sys.exit(0 if ok else 1)
