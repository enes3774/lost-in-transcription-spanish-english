"""Turn a folder of audio files into the data directory layout main.py reads (the competition's format).

usage:  python prepare_data.py <audio_folder> <data_dir>

Writes <data_dir>/clips/<file> (the audio copied as-is; wav/mp3/flac/ogg/m4a are all read by the stages) and
<data_dir>/test_metadata.csv with the columns audio_filename, file_duration_seconds, language.
Duration is read with soundfile (librosa as fallback); it only steers batching and the long-clip cutting,
so an approximate value is fine. Needs: soundfile (and librosa for mp3/m4a).
"""
import csv
import shutil
import sys
from pathlib import Path

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}


def duration(path):
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        import librosa
        return float(librosa.get_duration(path=str(path)))


src, dst = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO)
if not files:
    sys.exit(f"no audio files in {src}")
(dst / "clips").mkdir(parents=True, exist_ok=True)
with open(dst / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["audio_filename", "file_duration_seconds", "language"])
    for p in files:
        shutil.copy(p, dst / "clips" / p.name)
        w.writerow([p.name, f"{duration(p):.2f}", "enspa"])
print(f"{len(files)} clips -> {dst}")
