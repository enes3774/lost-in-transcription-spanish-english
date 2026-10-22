"""Score a Whisper decode of the selection half of T with runaway repetition collapsed. Prints one number."""
import csv
import sys

sys.path.insert(0, "/kaggle/working/omni_rt")
import jiwer  # noqa: E402
from score import normalize_text  # noqa: E402


def collapse(text, max_uni=4, max_ngram=2):
    w = text.split()
    for n in (6, 5, 4, 3, 2, 1):
        cap = max_uni if n == 1 else max_ngram
        out, i = [], 0
        while i < len(w):
            seg = [x.lower().strip(".,?!") for x in w[i:i + n]]
            reps = 1
            while len(seg) == n and [x.lower().strip(".,?!") for x in w[i + reps * n:i + (reps + 1) * n]] == seg:
                reps += 1
            out.extend(w[i:i + min(reps, cap) * n])
            i += reps * n
        w = out
    return " ".join(w)


if __name__ == "__main__":
    K = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold/miami_Tsel600/refs.csv"
    load = lambda p: {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(p, encoding="utf-8"))}
    refs, hy = load(K), load(sys.argv[1])
    names = [n for n in sorted(refs) if normalize_text(refs[n]).strip()]
    print(f"{jiwer.wer([normalize_text(refs[n]) for n in names], [normalize_text(collapse(hy.get(n, ''))) for n in names]):.4f}")
