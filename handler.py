import os, io, json, base64, tempfile, time, math
batch_size = int(p.get("batch_size", DEFAULT_BATCH))
compute_type = p.get("compute_type", DEFAULT_COMPUTE)


language = p.get("language") # e.g., "ru" to force RU
align = bool(p.get("align", True))
char_align = bool(p.get("char_align", False))
diarize = bool(p.get("diarize", False))
hf_token = p.get("hf_token") or HF_TOKEN_ENV


return_srt = bool(p.get("return_srt", True))
return_vtt = bool(p.get("return_vtt", False))


# optional whisper knobs
# condition_on_prev_text=False уменьшает галлюцинации, VAD включён в whisperx по-умолчанию
whisper_kwargs = {
"batch_size": batch_size,
}


# 1) load audio
audio_path = _download_to_tmp(p)
audio = whisperx.load_audio(audio_path)


# 2) ASR
asr = _ensure_model(model_name, compute_type, batch_size)
result = asr.transcribe(
audio,
**whisper_kwargs,
language=language,
)


# 3) Alignment (wav2vec2)
if align:
lang = language or result.get("language")
if not lang:
lang = "ru" # дефолт, если не распозналось
align_model, meta = _ensure_aligner(lang)
result = whisperx.align(
result["segments"], align_model, meta, audio, DEVICE,
return_char_alignments=char_align
)


# 4) (Optional) diarization
diarize_segments = None
if diarize:
if not hf_token:
raise ValueError("Diarization requested but no HF token provided (env HF_TOKEN or input.hf_token)")
diar = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
diarize_segments = diar(audio,
min_speakers=p.get("min_speakers"),
max_speakers=p.get("max_speakers"))
result = whisperx.assign_word_speakers(diarize_segments, result)


# 5) build outputs
segments = result.get("segments", result) # result может уже быть списком сегментов


out = {
"device": DEVICE,
"model": model_name,
"compute_type": compute_type,
"language": language or result.get("language"),
"duration_sec": float(p.get("duration_override", 0)) or None,
"segments": segments, # [{id, start, end, text, words: [{word,start,end,score}], speaker?}]
"diarization": diarize_segments,
"timing": {"total_sec": round(time.time() - t0, 3)}
}


if return_srt:
out["srt"] = _make_srt(segments)
if return_vtt:
out["vtt"] = _make_vtt(segments)


try:
os.remove(audio_path)
except Exception:
pass


return out




# Entrypoint
runpod.serverless.start({"handler": handler})