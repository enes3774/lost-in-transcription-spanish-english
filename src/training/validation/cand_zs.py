"""Zero-shot decode of a new candidate member on a competition data dir (clips/ + test_metadata.csv; long clips already
cut by seg_long). usage: cand_zs.py <backend> <model_dir> <data_dir> <out_csv> [batch]
backends: granite_nar (transformers>=5.5, no prompt / no language), phi4 (trust_remote_code, ASR prompt), moss.
Output text gets the same loop collapse as run_canary_dev.py. Nothing is caught: a failure stops the run."""
import csv
import sys
import time
from pathlib import Path

import librosa
import torch

BACK, MODEL, DATA, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
BS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
SR = 16000


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def collapse(text, max_uni=4, max_ngram=2):
    w = (text or "").split()
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


rows = list(csv.DictReader(open(DATA / "test_metadata.csv", encoding="utf-8")))
names = [r["audio_filename"] for r in rows]
dur = {r["audio_filename"]: float(r.get("file_duration_seconds") or 0) for r in rows}
order = sorted(names, key=lambda n: -dur.get(n, 0))
t0 = time.time()

if BACK == "granite_nar":
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, attn_implementation="sdpa", dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    def run(wavs, mx):
        inputs = processor([torch.from_numpy(w) for w in wavs], device="cuda")
        out = model.transcribe(**inputs)
        return processor.batch_decode(out.preds)
elif BACK == "phi4":
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 _attn_implementation="sdpa").to("cuda").eval()
    gcfg = GenerationConfig.from_pretrained(MODEL)
    import os
    if os.environ.get("PHI4_DELTA"):                  # Miami fine-tuned tensors from phi4_train.py
        from safetensors.torch import load_file
        dl = load_file(os.environ["PHI4_DELTA"])
        sd = dict(model.named_parameters())
        assert set(dl) <= set(sd), "delta names not in model"
        with torch.no_grad():
            for k, v in dl.items():
                sd[k].copy_(v.to(sd[k].dtype))
        log(f"PHI4_DELTA: {len(dl)} tensors loaded")
    PROMPT = "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>"
    processor.tokenizer.padding_side = "left"

    def run(wavs, mx):
        inputs = processor(text=[PROMPT] * len(wavs), audios=[(w, SR) for w in wavs], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            g = model.generate(**inputs, max_new_tokens=mx, generation_config=gcfg, do_sample=False, num_logits_to_keep=1)
        g = g[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
elif BACK == "gemma":                              # gemma-4-E4B-it; GEMMA_FT=<gemma_train.py best dir> for the Miami model
    import os
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = "left"
    model = Gemma4ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    if os.environ.get("GEMMA_FT"):
        from peft import PeftModel
        from safetensors.torch import load_file
        ft = os.environ["GEMMA_FT"]
        model = PeftModel.from_pretrained(model, f"{ft}/adapter").merge_and_unload().eval()
        au = load_file(f"{ft}/audio.safetensors")
        sd = dict(model.named_parameters())
        strip = lambda k: k.replace("base_model.model.", "", 1)
        assert all(strip(k) in sd for k in au), "audio tensor names not in model"
        with torch.no_grad():
            for k, v in au.items():
                sd[strip(k)].copy_(v.to(sd[strip(k)].dtype))
        log(f"GEMMA_FT: adapter merged + {len(au)} audio tensors from {ft}")
    GP = ("Transcribe this audio word for word. The speaker code-switches freely between English and Spanish, often "
          "inside a single sentence. Write every word in the language it was actually spoken in. Never translate "
          "between English and Spanish. Do not summarise, do not rephrase, do not fix grammar, do not add words that "
          "were not said. Output only the transcript itself, with no preamble, no quotes and no commentary.")

    def run(wavs, mx):
        convs = [[{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": GP}]}] for w in wavs]
        inp = processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt", padding=True, sampling_rate=SR)
        inp = {k: (v.to("cuda", torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda")) for k, v in inp.items()}
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=mx, do_sample=False)
        return processor.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
elif BACK == "moss":
    import re
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    # the model card's default (timestamped + diarized) prompt; tags are stripped from the output below
    MP = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
    ptxt = processor.apply_chat_template([{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": MP}]}],
                                         tokenize=False, add_generation_prompt=True)
    TAG = re.compile(r"\[\s*\d+(?:\.\d+)?\s*\]|\[S\d+\]")

    def run(wavs, mx):  # batch 1: the processor right-pads, which a decoder-only generate cannot batch
        res = []
        for w in wavs:
            inputs = processor(text=ptxt, audio=w, return_tensors="pt").to("cuda")
            inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
            with torch.inference_mode():
                g = model.generate(**inputs, max_new_tokens=mx + 64, do_sample=False)
            txt = processor.tokenizer.decode(g[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            res.append(" ".join(TAG.sub(" ", txt).split()))
        return res
elif BACK == "seamless":                           # SeamlessM4T-v2 (seamless_spen_train.py best dir), target language spa
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    processor = AutoProcessor.from_pretrained(MODEL)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

    def run(wavs, mx):
        x = processor(audios=wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        x = {k: v.cuda() for k, v in x.items()}
        with torch.inference_mode():
            y = model.generate(**x, tgt_lang="spa", num_beams=1, max_new_tokens=min(mx, 256))
        return processor.batch_decode(y, skip_special_tokens=True)
elif BACK == "w2vbert":                            # w2v-BERT 2.0 CTC (w2vbert_train.py best dir), greedy
    from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
    processor = Wav2Vec2BertProcessor.from_pretrained(MODEL)
    model = Wav2Vec2BertForCTC.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

    def run(wavs, mx):
        x = processor.feature_extractor(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.inference_mode():
            lg = model(input_features=x["input_features"].to("cuda", torch.bfloat16), attention_mask=x["attention_mask"].cuda()).logits
        return processor.tokenizer.batch_decode(lg.argmax(-1).cpu())
elif BACK == "vibevoice":                          # VibeVoice-ASR 8.3B, 24 kHz; VIBE_FT=<vibevoice_train.py adapter dir>
    import json
    import os
    import re
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
    SR = 24000
    processor = AutoProcessor.from_pretrained(MODEL)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    if os.environ.get("VIBE_FT"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.environ["VIBE_FT"]).merge_and_unload().eval()
        log(f"VIBE_FT: adapter merged from {os.environ['VIBE_FT']}")
    CONTENT = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)"')
    SPECIAL = re.compile(r"<\|[^|]*\|>")

    def parse(raw):                                 # same reader as vibevoice_train.py: JSON Content fields, else raw text
        txt = SPECIAL.sub(" ", raw)
        parts = CONTENT.findall(txt)
        if parts:
            out = []
            for p in parts:
                try:
                    out.append(json.loads('"' + p + '"'))
                except json.JSONDecodeError:
                    out.append(p.replace('\\"', '"'))
            return " ".join(s.strip() for s in out if s.strip())
        return re.sub(r"\s+", " ", txt.replace("assistant", " ", 1)).strip()

    def run(wavs, mx):
        enc = processor.apply_transcription_request(audio=wavs, sampling_rate=SR).to(model.device, model.dtype)
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=max(mx, 128), do_sample=False)
        raws = processor.decode(o[:, enc["input_ids"].shape[1]:])
        raws = [raws] if isinstance(raws, str) else raws
        return [parse(str(r)) for r in raws]
elif BACK == "voxtral":                            # Voxtral (HF format) + optional LoRA (VOX_FT=<kf_voxtral_mini.py best dir>)
    import os
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import soundfile as _sf
    import tempfile
    processor = AutoProcessor.from_pretrained(MODEL)
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    if os.environ.get("VOX_FT"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.environ["VOX_FT"]).merge_and_unload().eval()
        log(f"VOX_FT: adapter merged from {os.environ['VOX_FT']}")
    VP = ("Transcribe this audio word for word.\n"      # the training prompt of kf_voxtral.py, verbatim
          "The speaker code-switches freely between English and Spanish, often inside a single "
          "sentence. Write every word in the language it was actually spoken in. "
          "Never translate between English and Spanish: if a phrase is spoken in English, "
          "keep it in English; if it is spoken in Spanish, keep it in Spanish.\n"
          "Do not summarise, do not rephrase, do not fix grammar, do not add words that were not said.\n"
          "Output only the transcript itself, with no preamble, no quotes and no commentary.")
    TMPD = tempfile.mkdtemp(prefix="voxdec_", dir="/dev/shm")

    def run(wavs, mx):
        paths = []
        for j, w in enumerate(wavs):
            p = f"{TMPD}/{j}.wav"
            _sf.write(p, w, SR)
            paths.append(p)
        convs = [[{"role": "user", "content": [{"type": "audio", "path": p}, {"type": "text", "text": VP}]}] for p in paths]
        x = processor.apply_chat_template(convs, return_tensors="pt")
        x = {k: v.to("cuda", torch.bfloat16 if v.dtype.is_floating_point else None) for k, v in x.items()}
        with torch.inference_mode():
            out = model.generate(**x, max_new_tokens=max(mx, 64), do_sample=False)
        return processor.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
elif BACK == "nemotron":                           # Nemotron 3.5 ASR RNNT (nemotron_train.py best dir), auto language
    import re
    sys.path.insert(0, "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/vendor_tf")
    from transformers import AutoModelForRNNT, AutoProcessor
    processor = AutoProcessor.from_pretrained("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/nemotron/streaming-0.6b")
    model = AutoModelForRNNT.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

    def run(wavs, mx):
        enc = processor(wavs, language="auto", sampling_rate=SR, return_tensors="pt", padding=True)
        enc = {k: (v.to("cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda")) for k, v in enc.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            seq = model.generate(**enc, return_dict_in_generate=True).sequences
        return [re.sub(r"<[^>]+>", " ", str(t)) for t in processor.batch_decode(seq, skip_special_tokens=True)]
else:
    raise SystemExit(f"unknown backend {BACK}")
log(f"{BACK} loaded in {time.time() - t0:.0f}s, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

preds, n_loop, t = {}, 0, time.time()
for i in range(0, len(order), BS):
    chunk = order[i:i + BS]
    wavs = [librosa.load(str(DATA / "clips" / n), sr=SR, mono=True)[0] for n in chunk]
    mx = max(64, int(max(dur.get(n, 0) for n in chunk) * 12) + 64)
    texts = run(wavs, mx)
    assert len(texts) == len(chunk)
    for n, x in zip(chunk, texts):
        raw = " ".join(str(x).split())
        c = collapse(raw)
        n_loop += len(raw.split()) > 2 * len(c.split()) + 3
        preds[n] = c
    if (i // BS) % 20 == 0:
        log(f"{min(i + BS, len(order))}/{len(order)} clips, {time.time() - t:.0f}s, {n_loop} looping")
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["audio_filename", "transcript"])
    for n in names:
        w.writerow([n, preds.get(n, "")])
el = time.time() - t
log(f"done: {len(preds)} clips in {el:.0f}s ({sum(dur.values()) / max(el, 1):.0f}x realtime), {n_loop} looping, "
    f"{sum(1 for v in preds.values() if not v.strip())} empty -> {OUT}")

