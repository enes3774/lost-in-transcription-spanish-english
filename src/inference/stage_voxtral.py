"""Voxtral-Small-24B (Miami-merged) transcription stage. vLLM first, HF transformers fallback.
usage: stage_voxtral.py <model_dir> <data_dir> <out_csv> [gpu_util]
"""
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROMPT, SR, clean_prediction, load_audio, log, read_manifest, write_csv  # noqa: E402


def main():
    model_dir, data_dir, out_csv = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    gpu_util = float(sys.argv[4]) if len(sys.argv) > 4 else 0.90
    rows = read_manifest(data_dir)
    order = sorted(rows, key=lambda r: -float(r.get("file_duration_seconds") or 0))
    pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
    preds = {}
    write_csv(out_csv, rows, preds)


    def run_vllm():
        from mistral_common.audio import Audio
        from mistral_common.protocol.instruct.messages import UserMessage
        try:
            from mistral_common.protocol.instruct.chunk import AudioChunk, RawAudio, TextChunk
        except ImportError:
            from mistral_common.protocol.instruct.messages import AudioChunk, RawAudio, TextChunk
        from vllm import LLM, SamplingParams

        # model_dir holds the mistral-format base (params.json, tekken.json, consolidated.safetensors);
        # the Miami LoRA (peft, HF naming) sits in model_dir/lora and is applied by vLLM at runtime.
        lora_dir = Path(model_dir) / "lora"
        lora_req = None
        kwargs = {}
        if (lora_dir / "adapter_config.json").exists():
            from vllm.lora.request import LoRARequest
            lora_req = LoRARequest("miami", 1, str(lora_dir))
            kwargs = dict(enable_lora=True, max_lora_rank=16, max_loras=1)
        llm = LLM(model=model_dir, tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
                  dtype="bfloat16", max_model_len=8192, max_num_seqs=128, gpu_memory_utilization=gpu_util,
                  limit_mm_per_prompt={"audio": 1}, seed=0, **kwargs)
        log(f"voxtral vllm engine ready (lora={'yes' if lora_req else 'no'})")
        chunk = 256
        for ci in range(0, len(order), chunk):
            part = order[ci:ci + chunk]
            wavs = list(pool.map(lambda r: load_audio(data_dir / "clips" / r["audio_filename"]), part))
            convs, params = [], []
            for r, wav in zip(part, wavs):
                audio = Audio(audio_array=wav, sampling_rate=SR, format="wav")
                convs.append([UserMessage(content=[AudioChunk(input_audio=RawAudio.from_audio(audio)), TextChunk(text=PROMPT)]).to_openai()])
                params.append(SamplingParams(temperature=0.0, max_tokens=max(32, int(len(wav) / SR * 20) + 64)))
            outs = llm.chat(convs, sampling_params=params, use_tqdm=False, lora_request=lora_req)
            for r, o in zip(part, outs):
                preds[r["audio_filename"]] = clean_prediction(o.outputs[0].text)
            write_csv(out_csv, rows, preds)
            log(f"voxtral chunk {ci // chunk + 1}/{(len(order) + chunk - 1) // chunk} done")


    def run_hf(batch=8):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        hf_dir = os.environ.get("VOX_HF_DIR", str(Path(model_dir) / "hf"))  # merged HF checkpoint (optional fallback)
        processor = AutoProcessor.from_pretrained(hf_dir)
        model = VoxtralForConditionalGeneration.from_pretrained(hf_dir, dtype=torch.bfloat16, device_map={"": 0}).eval()
        log("voxtral hf model ready")
        for i in range(0, len(order), batch):
            part = order[i:i + batch]
            conv = [[{"role": "user", "content": [{"type": "audio", "path": str(data_dir / "clips" / r["audio_filename"])}, {"type": "text", "text": PROMPT}]}] for r in part]
            x = processor.apply_chat_template(conv, return_tensors="pt")
            x = {k: v.to("cuda", torch.bfloat16 if v.dtype.is_floating_point else None) for k, v in x.items()}
            maxd = max(float(r.get("file_duration_seconds") or 30) for r in part)
            with torch.no_grad():
                out = model.generate(**x, max_new_tokens=max(32, int(maxd * 20) + 64), do_sample=False, num_beams=1)
            for r, t in zip(part, processor.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)):
                preds[r["audio_filename"]] = clean_prediction(t)
            if (i // batch) % 20 == 0:
                write_csv(out_csv, rows, preds)
                log(f"voxtral hf batch {i // batch + 1}")


    t0 = time.time()
    if os.environ.get("VOX_BACKEND", "vllm") == "vllm":
        run_vllm()                      # STRICT_V8: a vLLM failure fails the stage (no HF fallback)
    else:
        run_hf()
    write_csv(out_csv, rows, preds)
    log(f"voxtral stage done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
