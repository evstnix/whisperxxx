# /src/handler.py
# WhisperX worker for RunPod Serverless (lazy imports, word-level align, optional diarization)

import os, json, base64, tempfile, time, math
import runpod
import requests

# -------- lazy heavy imports (ускоряет cold start) --------
_torch = None
_whisperx = None
def _lazy_imports():
    global _torch, _whisperx
    if _torch is None:
        import torch as _t
        _torch = _t
    if _whisperx is None:
        import whisperx as _w
        _whisperx = _w
    return _torch, _whisperx

# -------- config / env --------
FORCE_DEVICE   = os.getenv("FORCE_DEVICE", "auto")  # auto|cuda|cpu
DEFAULT_MODEL  = os.getenv("MODEL_NAME", "large-v3")
DEFAULT_BATCH  = int(os.getenv("BATCH_SIZE", "8"))
DEFAULT_COMPUTE= os.getenv("COMPUTE_TYPE", "float16")
HF_TOKEN_ENV   = os.getenv("HF_TOKEN")

ALLOWED_WHISPER_ARGS = {
    "beam_size","patience","length_penalty","temperature",
    "compression_ratio_threshold","log_prob_threshold","no_speech_threshold",
    "condition_on_previous_text","initial_prompt","vad_filter","vad_parameters",
    "prompt_reset_on_temperature","temperature_increment_on_fallback",
    "without_timestamps","word_timestamps","chunk_size"
}

# -------- caches --------
_model = None
_model_cfg = {}
_align_model = None
_align_meta  = None
_align_cfg   = {}

# -------- helpers --------
def _device():
    t, _ = _lazy_imports()
    if FORCE_DEVICE == "cuda" and t.cuda.is_available():
        return "cuda"
    if FORCE_DEVICE == "cpu":
        return "cpu"
    return "cuda" if t.cuda.is_available() else "cpu"

def _normalize_compute_type(device, requested):
    if device == "cpu":
        return "int8"  # безопасный дефолт на CPU
    return requested or "float16"

def _download_to_tmp(p):
    if p.get("audio_url"):
        url = p["audio_url"]
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            suffix = os.path.splitext(url.split("?")[0])[1] or ".wav"
            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return path
    if p.get("audio_b64"):
        raw = base64.b64decode(p["audio_b64"])
        fd, path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path
    if p.get("audio_path"):
        return p["audio_path"]
    raise ValueError("Provide 'audio_url' or 'audio_b64' or 'audio_path'.")

def _ensure_model(model_name, compute_type, batch_size):
    global _model, _model_cfg
    _, wx = _lazy_imports()
    dev = _device()
    ctype = _normalize_compute_type(dev, compute_type)
    if _model is None or _model_cfg.get("name") != model_name or _model_cfg.get("ctype") != ctype:
        _model = wx.load_model(model_name, dev, compute_type=ctype)
        _model_cfg = {"name": model_name, "ctype": ctype}
    return _model

def _ensure_aligner(lang_code, model_name=None):
    global _align_model, _align_meta, _align_cfg
    _, wx = _lazy_imports()
    dev = _device()
    if (
        _align_model is None
        or _align_meta is None
        or _align_cfg.get("lang") != lang_code
        or (model_name and _align_cfg.get("model_name") != model_name)
    ):
        _align_model, _align_meta = wx.load_align_model(
            language_code=lang_code, device=dev, model_name=model_name
        )
        _align_cfg = {"lang": lang_code, "model_name": model_name}
    return _align_model, _align_meta

def _ts_srt(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _make_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(seg.get('start', 0))} --> {_ts_srt(seg.get('end', 0))}")
        lines.append((seg.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines)

def _make_vtt(segments):
    def ts(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{ts(seg.get('start',0))} --> {ts(seg.get('end',0))}")
        lines.append((seg.get("text") or "").strip()); lines.append("")
    return "\n".join(lines)

# -------- main handler --------
def handler(job):
    t0 = time.time()
    p = job.get("input", {})

    model_name   = p.get("model", DEFAULT_MODEL)
    batch_size   = int(p.get("batch_size", DEFAULT_BATCH))
    compute_type = p.get("compute_type", DEFAULT_COMPUTE)

    language   = p.get("language")          # "ru" и т.п.
    align      = bool(p.get("align", True))
    char_align = bool(p.get("char_align", False))
    diarize    = bool(p.get("diarize", False))
    hf_token   = p.get("hf_token") or HF_TOKEN_ENV

    align_model_name = p.get("align_model")  # явный HF aligner
    return_raw = bool(p.get("return_raw", True))
    return_srt = bool(p.get("return_srt", True))
    return_vtt = bool(p.get("return_vtt", False))

    whisper_over   = p.get("whisper", {}) or {}
    whisper_kwargs = {"batch_size": batch_size}
    for k, v in whisper_over.items():
        if k in ALLOWED_WHISPER_ARGS:
            whisper_kwargs[k] = v
    whisper_kwargs.setdefault("condition_on_previous_text", False)
    whisper_kwargs.setdefault("vad_filter", True)

    # 1) load audio
    _, wx = _lazy_imports()
    audio_path = _download_to_tmp(p)
    audio = wx.load_audio(audio_path)

    # 2) ASR
    asr = _ensure_model(model_name, compute_type, batch_size)
    result_asr = asr.transcribe(audio, language=language, **whisper_kwargs)

    segments_raw  = result_asr.get("segments", [])
    detected_lang = result_asr.get("language") or language

    # 3) Alignment (wav2vec2)
    segments_aligned = segments_raw
    diarize_segments = None
    if align and segments_raw:
        lang = (language or detected_lang or "ru")
        align_model, meta = _ensure_aligner(lang, model_name=align_model_name)
        aligned = wx.align(segments_raw, align_model, meta, audio, _device(),
                           return_char_alignments=char_align)
        segments_aligned = aligned.get("segments", aligned)

    # 4) (Optional) diarization
    if diarize:
        if not hf_token:
            raise ValueError("Diarization requested but no HF token provided (env HF_TOKEN or input.hf_token)")
        diar = wx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=_device())
        diarize_segments = diar(audio,
                                min_speakers=p.get("min_speakers"),
                                max_speakers=p.get("max_speakers"))
        segments_aligned = wx.assign_word_speakers(diarize_segments, {"segments": segments_aligned})["segments"]

    # 5) build output
    out = {
        "device": _device(),
        "model": model_name,
        "compute_type": _model_cfg.get("ctype"),
        "language": detected_lang,
        "timing": {"total_sec": round(time.time() - t0, 3)}
    }
    if return_raw:
        out["segments_raw"] = segments_raw
    out["segments"] = segments_aligned
    if diarize_segments is not None:
        out["diarization"] = diarize_segments
    if return_srt:
        out["srt"] = _make_srt(segments_aligned)
    if return_vtt:
        out["vtt"] = _make_vtt(segments_aligned)

    try:
        os.remove(audio_path)
    except Exception:
        pass

    return out

# -------- runpod entry --------
runpod.serverless.start({"handler": handler})
