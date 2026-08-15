# -*- coding: utf-8 -*-
"""即時字幕(Phase 1):雙路分軌收音 → X-ASR 串流辨識 → console 字幕 + 落盤。

- 麥克風一軌(本地)、WASAPI loopback 一軌(遠端,線上會議對方聲音),分軌不混音:
  時鐘漂移退化為時間戳誤差、避開回音重複轉錄、免費得到本地/遠端說話人分界。
- 每軌原生取樣率餵入(sherpa-onnx 內部重採樣),endpoint 偵測斷句,OpenCC s2twp 轉台灣正體。
- 每軌同時錄成 WAV(mono int16),會後可餵既有批次管線出權威版逐字稿。
- 產物在 session 目錄:subtitles.jsonl(定稿行)、mic.wav / loopback.wav。

用法:
  python live_subtitles.py                     # 雙軌
  python live_subtitles.py --mic-only          # 實體會議(只收麥克風)
  python live_subtitles.py --loopback-only     # 只收系統音
  python live_subtitles.py --list-devices
  python live_subtitles.py --hotwords hotwords.txt   # 熱詞(改用 modified_beam_search)
  python live_subtitles.py --duration 70       # 跑固定秒數(測試用),預設 Ctrl+C 結束
"""
import argparse
import datetime as dt
import json
import os
import queue
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Windows 無系統 CUDA 時,把 pip 版 CUDA DLL 加進搜尋路徑(同 pipeline.py,須在 import sherpa_onnx 前)
_nvidia_base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for _b in sorted(_nvidia_base.glob("*/bin")):
    os.add_dll_directory(str(_b))
    os.environ["PATH"] = str(_b) + os.pathsep + os.environ["PATH"]

BASE = Path(os.environ["LOCALAPPDATA"]) / "meeting-minutes"
DEFAULT_MODEL = BASE / "models" / \
    "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"
CHUNK_MS = 100


def ensure_model(model_dir: Path):
    """模型不在就從 sherpa-onnx release 抓(~128MB),冪等。"""
    if (model_dir / "tokens.txt").exists():
        return
    import tarfile
    import urllib.request
    url = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
           f"{model_dir.name}.tar.bz2")
    print(f"下載模型 {model_dir.name} ...", flush=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_dir.parent / (model_dir.name + ".tar.bz2")
    urllib.request.urlretrieve(url, tmp)
    with tarfile.open(tmp, "r:bz2") as tf:
        tf.extractall(model_dir.parent)
    tmp.unlink()


def load_recognizer(model_dir: Path, hotwords: str | None, provider: str):
    import sherpa_onnx

    def pick(stem):
        cands = sorted(model_dir.glob(f"{stem}*.onnx"))
        int8 = [p for p in cands if "int8" in p.name]
        return str(int8[0] if int8 else cands[0])

    kw = dict(
        tokens=str(model_dir / "tokens.txt"),
        encoder=pick("encoder"),
        decoder=pick("decoder"),
        joiner=pick("joiner"),
        num_threads=8,
        sample_rate=16000,
        feature_dim=80,
        provider=provider,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.0,
        rule3_min_utterance_length=18.0,
        decoding_method="greedy_search",
    )
    if hotwords:
        kw.update(
            decoding_method="modified_beam_search",
            hotwords_file=hotwords,
            hotwords_score=1.5,
            modeling_unit="cjkchar+bpe",
            bpe_vocab=str(model_dir / "bpe.model"),
        )
    return sherpa_onnx.OnlineRecognizer.from_transducer(**kw)


class Track:
    def __init__(self, name, label, device_index, p, rec, session: Path):
        info = p.get_device_info_by_index(device_index)
        self.name, self.label = name, label
        self.sr = int(info["defaultSampleRate"])
        self.ch = max(1, int(info["maxInputChannels"]))
        self.q = queue.Queue()
        self.stream_s = rec.create_stream()
        self.wav = wave.open(str(session / f"{name}.wav"), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(self.sr)
        self.utt_start = None  # 該行第一個非空 partial 的掛鐘時間
        self.last_data = time.time()
        self.pa_stream = p.open(
            format=8,  # paInt16
            channels=self.ch,
            rate=self.sr,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=int(self.sr * CHUNK_MS / 1000),
            stream_callback=self._cb,
        )
        print(f"  [{label}] {info['name']} @{self.sr}Hz ch{self.ch}")

    def _cb(self, in_data, frame_count, time_info, status):
        self.q.put(in_data)
        return (None, 0)

    def drain(self):
        """收 callback 送來的資料:寫 WAV、餵辨識 stream。回傳是否有新資料。"""
        got = False
        while True:
            try:
                raw = self.q.get_nowait()
            except queue.Empty:
                return got
            got = True
            self.last_data = time.time()
            pcm = np.frombuffer(raw, dtype=np.int16)
            if self.ch > 1:
                pcm = pcm.reshape(-1, self.ch).mean(axis=1).astype(np.int16)
            self.wav.writeframes(pcm.tobytes())
            self.stream_s.accept_waveform(self.sr, pcm.astype(np.float32) / 32768.0)

    def keepalive(self):
        """WASAPI loopback 無播放時不送資料 → 餵零樣本補靜音,endpoint 才會觸發、WAV 時間軸才對得上掛鐘。"""
        gap = time.time() - self.last_data
        if gap > 0.25:
            n = int(self.sr * gap)
            self.wav.writeframes(b"\x00\x00" * n)
            self.stream_s.accept_waveform(self.sr, np.zeros(n, dtype=np.float32))
            self.last_data = time.time()

    def close(self):
        try:
            self.pa_stream.stop_stream()
            self.pa_stream.close()
        finally:
            self.wav.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--mic-only", action="store_true")
    ap.add_argument("--loopback-only", action="store_true")
    ap.add_argument("--mic-index", type=int)
    ap.add_argument("--loopback-index", type=int)
    ap.add_argument("--hotwords")
    ap.add_argument("--terms", help="術語顯示映射檔(預設用腳本旁的 terms.txt)")
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--duration", type=float, help="跑固定秒數後自動結束(測試用)")
    ap.add_argument("--session-dir")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    import pyaudiowpatch as pyaudio
    from opencc import OpenCC

    p = pyaudio.PyAudio()
    if args.list_devices:
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["maxInputChannels"] > 0:
                print(f"{i:3d} {d['name']} @{int(d['defaultSampleRate'])}Hz ch{d['maxInputChannels']}")
        return

    session = Path(args.session_dir) if args.session_dir else \
        BASE / "live" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)

    ensure_model(Path(args.model))
    print(f"載入模型 {Path(args.model).name} ...", flush=True)
    try:
        rec = load_recognizer(Path(args.model), args.hotwords, args.provider)
        print(f"provider={args.provider}", flush=True)
    except RuntimeError as e:
        if args.provider == "cuda":
            print(f"CUDA 不可用({e}),退回 CPU", flush=True)
            rec = load_recognizer(Path(args.model), args.hotwords, "cpu")
        else:
            raise
    warm = rec.create_stream()  # 先暖機(CUDA 首次推論慢),再開收音
    warm.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
    while rec.is_ready(warm):
        rec.decode_stream(warm)
    s2twp = OpenCC("s2twp")

    terms_path = Path(args.terms) if args.terms else Path(__file__).parent / "terms.txt"
    terms = []
    if terms_path.exists():
        for ln in terms_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                terms.append(tuple(ln.split("=", 1)))
        terms.sort(key=lambda kv: -len(kv[0]))

    def to_display(text):
        out = s2twp.convert(text)
        for a, b in terms:
            out = out.replace(a, b)
        return out

    tracks = []
    if not args.loopback_only:
        idx = args.mic_index if args.mic_index is not None else p.get_default_input_device_info()["index"]
        tracks.append(Track("mic", "本地", idx, p, rec, session))
    if not args.mic_only:
        idx = args.loopback_index if args.loopback_index is not None else p.get_default_wasapi_loopback()["index"]
        tracks.append(Track("loopback", "遠端", idx, p, rec, session))
    print(f"session: {session}\n--- 開始(Ctrl+C 結束)---", flush=True)

    jsonl = open(session / "subtitles.jsonl", "a", encoding="utf-8")
    partial_shown = ""

    def emit(tr, text):
        nonlocal partial_shown
        raw = s2twp.convert(text.strip())
        line = to_display(text.strip())
        ts = dt.datetime.fromtimestamp(tr.utt_start or time.time())
        sys.stdout.write("\r" + " " * len(partial_shown.encode("gbk", "replace")) + "\r")
        partial_shown = ""
        print(f"[{ts:%H:%M:%S}][{tr.label}] {line}", flush=True)
        rec_line = {"t": ts.isoformat(timespec="seconds"), "track": tr.name,
                    "label": tr.label, "text": line}
        if line != raw:
            rec_line["raw"] = raw
        jsonl.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
        jsonl.flush()

    t0 = time.time()
    try:
        while args.duration is None or time.time() - t0 < args.duration:
            for tr in tracks:
                tr.drain()
                tr.keepalive()
            active = [tr for tr in tracks if rec.is_ready(tr.stream_s)]
            if active:
                rec.decode_streams([tr.stream_s for tr in active])
            for tr in tracks:
                text = rec.get_result(tr.stream_s)
                if text and tr.utt_start is None:
                    tr.utt_start = time.time()
                if rec.is_endpoint(tr.stream_s):
                    if text.strip():
                        emit(tr, text)
                    rec.reset(tr.stream_s)
                    tr.utt_start = None
            # 部分結果顯示:取最近開口的那軌
            talking = [tr for tr in tracks if rec.get_result(tr.stream_s)]
            show = ""
            if talking:
                tr = max(talking, key=lambda t: t.utt_start or 0)
                show = f"… [{tr.label}] {to_display(rec.get_result(tr.stream_s))}"[-80:]
            if show != partial_shown:
                pad = max(0, len(partial_shown) - len(show))
                sys.stdout.write("\r" + show + " " * pad)
                sys.stdout.flush()
                partial_shown = show
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        for tr in tracks:  # 沖出未定稿的殘句
            text = rec.get_result(tr.stream_s)
            if text.strip():
                emit(tr, text)
        print(f"\n--- 結束,產物在 {session} ---")
        for tr in tracks:
            tr.close()
        jsonl.close()
        p.terminate()


if __name__ == "__main__":
    main()
