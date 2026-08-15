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


def load_terms(path: Path):
    """讀術語顯示映射檔(錯=對,一行一組),長詞優先。refiner 也共用。"""
    terms = []
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                terms.append(tuple(ln.split("=", 1)))
    terms.sort(key=lambda kv: -len(kv[0]))
    return terms


def align_final(commit, line):
    """final 與已顯示前綴的對齊:回傳 (stable, replace_last, missed_tail)。
    - commit 是 line 的前綴 → 未顯示 suffix 全量原子補上(不限字數,是新字不是回改)
    - 分歧只在尾端 ≤12 字 → 允許替換該尾端(規格唯一允許的回改範圍)
    - 分歧更深 → 用 commit 尾錨在 final 中找續文補尾;找不到才記 missed_tail
    """
    if line.startswith(commit):
        return line, 0, False
    n = 0
    for a, b in zip(commit, line):
        if a != b:
            break
        n += 1
    if len(commit) - n <= 12:
        return line, len(commit) - n, False
    tail_alnum = "".join(ch for ch in commit if ch.isalnum())
    for klen in (8, 6, 4):  # 錨長漸退,提高補尾命中率
        key = tail_alnum[-klen:]
        if len(key) < klen:
            continue
        acc = ""
        pos = None
        for i, ch in enumerate(line):
            if ch.isalnum():
                acc += ch
                if acc.endswith(key):
                    pos = i + 1
        if pos is not None and pos < len(line):
            return commit + line[pos:], 0, False
    return commit, 0, True


def apply_terms(text, terms):
    import re
    for a, b in terms:
        if a.isascii():  # 英文術語用字界比對,避免誤傷含該子字串的單字(ob vs problem)
            text = re.sub(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", b, text)
        else:
            text = text.replace(a, b)
    return text


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
        rule2_min_trailing_silence=0.8,  # 停頓 0.8s 定稿:0.6 會切出太多碎片害 2-pass 幻覺,1.0 又拖延遲
        rule3_min_utterance_length=18.0,
        decoding_method="greedy_search",
    )
    if hotwords:
        # 實測:此模型中文為 BPE pieces,modeling_unit 必須用 "bpe"(cjkchar 查表會全滅);
        # 需 bpe.vocab(用 sentencepiece 從 bpe.model 匯出)。A/B 實測僅 -0.5pp CER,故不預設開
        kw.update(
            decoding_method="modified_beam_search",
            hotwords_file=hotwords,
            hotwords_score=1.5,
            modeling_unit="bpe",
            bpe_vocab=str(model_dir / "bpe.vocab"),
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
        self.utt_start_sample = None  # 同上,但記樣本位移(供 2-pass 切片)
        self.utt_id = None  # 全域句編號
        # LocalAgreement:兩輪取樣一致的前綴才提交,已提交的字永不回改
        self.la_last = ""  # 上次取樣的完整假設
        self.la_commit = ""  # 已提交前綴
        self.la_t = 0.0  # 上次取樣時間
        self.last_loud = 0.0  # 此軌最近一次有聲音的時間(雙音源仲裁用)
        self.recent_finals = []  # [(start_epoch, line)] 近幾句定稿(跨軌 bleed 比對用)
        self.suspect = False  # 本句疑似回授(live 階段即攔,不等 final)
        self.total = 0  # 累計餵入樣本數(含 keepalive 靜音)
        self.buf = []  # 滾動 PCM 緩衝(mono int16 陣列串),供 2-pass 回切句音訊
        self.buf_start = 0  # buf[0] 開頭對應的樣本位移
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
            self._ingest(pcm)

    def keepalive(self):
        """WASAPI loopback 無播放時不送資料 → 餵零樣本補靜音,endpoint 才會觸發、WAV 時間軸才對得上掛鐘。"""
        gap = time.time() - self.last_data
        if gap > 0.25:
            self._ingest(np.zeros(int(self.sr * gap), dtype=np.int16))
            self.last_data = time.time()

    def _ingest(self, pcm):
        if pcm.size and float(np.abs(pcm.astype(np.int32)).mean()) > 80:
            self.last_loud = time.time()
        self.wav.writeframes(pcm.tobytes())
        self.stream_s.accept_waveform(self.sr, pcm.astype(np.float32) / 32768.0)
        self.total += len(pcm)
        self.buf.append(pcm)
        while self.total - self.buf_start > 40 * self.sr:  # 只留最近 40s
            self.buf_start += len(self.buf.pop(0))

    def slice_from(self, start_sample):
        """取 [start_sample, 現在] 的 mono int16(供 2-pass 重跑該句)。"""
        start = max(start_sample, self.buf_start)
        out, pos = [], self.buf_start
        for a in self.buf:
            end = pos + len(a)
            if end > start:
                out.append(a[max(0, start - pos):])
            pos = end
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)

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
    ap.add_argument("--no-refine", action="store_true",
                    help="停用句尾 Breeze 2-pass 修正(預設啟用,另開 refiner 子程序)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="停用電影字幕式螢幕疊加層(預設啟用)")
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

    # 熱詞:未指定時用腳本旁的 hotwords.txt(要 bpe.vocab 才能編碼,無則跳過)
    if not args.hotwords:
        hp = Path(__file__).parent / "hotwords.txt"
        args.hotwords = str(hp) if hp.exists() else None
    if args.hotwords and not (Path(args.model) / "bpe.vocab").exists():
        print("hotwords 需要模型目錄有 bpe.vocab(sentencepiece 自 bpe.model 匯出),本次停用", flush=True)
        args.hotwords = None

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
    terms = load_terms(terms_path)

    import re as _re
    _letters = _re.compile(r"\b(?:[A-Za-z]\s)+[A-Za-z]\b")  # 單字母序列:V I P→VIP、L A→LA
    _decimal = _re.compile(r"(\d)\s*\.\s*(\d)")  # 0 . 5→0.5
    # 中文小數(一點二→1.2):窄模式,前後不接其他數量詞才轉,避免誤傷「有一點難」
    _zhnum = "零一二三四五六七八九"
    _zhdec = _re.compile(rf"(?<![{_zhnum}十百千])([{_zhnum}])點([{_zhnum}]{{1,4}})(?![{_zhnum}十百千萬分])")

    def to_display(text):
        t = s2twp.convert(text)
        t = _letters.sub(lambda m: m.group(0).replace(" ", ""), t)
        t = _decimal.sub(r"\1.\2", t)
        t = _zhdec.sub(lambda m: f"{_zhnum.index(m.group(1))}."
                       + "".join(str(_zhnum.index(c)) for c in m.group(2)), t)
        return apply_terms(t, terms)

    tracks = []
    if not args.loopback_only:
        idx = args.mic_index if args.mic_index is not None else p.get_default_input_device_info()["index"]
        tracks.append(Track("mic", "本地", idx, p, rec, session))
    if not args.mic_only:
        idx = args.loopback_index if args.loopback_index is not None else p.get_default_wasapi_loopback()["index"]
        tracks.append(Track("loopback", "遠端", idx, p, rec, session))
    print(f"session: {session}\n--- 開始(Ctrl+C 結束)---", flush=True)

    import subprocess
    import threading

    # UI render event log:每個顯示決策(提交/定稿/丟棄與原因)落盤,回放對照抓回跳用
    render_log = open(session / "render_log.jsonl", "a", encoding="utf-8")

    def rlog(ev):
        ev["w"] = round(time.time(), 3)
        render_log.write(json.dumps(ev, ensure_ascii=False) + "\n")
        render_log.flush()

    # 電影字幕式螢幕疊加層(只有文字浮在畫面上,滑鼠穿透)
    overlay_proc = None
    ov_lock = threading.Lock()
    if not args.no_overlay:
        overlay_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "overlay.py")],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    def ov_send(ev):
        if overlay_proc is None:
            return
        try:
            with ov_lock:  # emit(主執行緒)與 refiner pump(子執行緒)都會寫
                overlay_proc.stdin.write(json.dumps(ev, ensure_ascii=False) + "\n")
                overlay_proc.stdin.flush()
        except OSError:
            pass

    # 句尾 2-pass:refiner 子程序吃 refine_queue 裡的句音訊,出修正行(⟳ 前綴)
    refine_dir = None
    refiner_proc = None
    refined_q = queue.Queue()
    if not args.no_refine:
        refine_dir = session / "refine_queue"
        refine_dir.mkdir(exist_ok=True)
        refiner_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "refiner.py"), str(session),
             "--terms", str(terms_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})

        def pump():
            for ln in refiner_proc.stdout:
                ln = ln.rstrip("\n")
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError:
                    refined_q.put(ln)
                    continue
                if ev.get("kind") == "offline":
                    # 離線校正只進紀錄與 console,不上畫面(顯示層=LA 穩定直播字幕,永不回改)
                    refined_q.put(f"⟳offline+{ev.get('lag_s', '?')}s[{ev['t'][11:]}][{ev['label']}] {ev['text']}")
                    rlog({"ev": "offline", "utt": ev.get("utt"), "seq": ev.get("seq"),
                          "lag_s": ev.get("lag_s"), "applied": "record_only"})
                elif ev.get("kind") == "status":
                    refined_q.put(ev["msg"])
                else:
                    refined_q.put(ln)
        threading.Thread(target=pump, daemon=True).start()

    jsonl = open(session / "subtitles.jsonl", "a", encoding="utf-8")
    partial_shown = ""
    seq = 0
    records = []  # 收尾時依 audio_start 排序輸出(兩軌同時完成會寫入倒置)

    FILLER_CHARS = set("呃嗯啊哦欸喔嘿哎唉就然後這些那個")
    lb_tr = next((t for t in tracks if t.name == "loopback"), None)

    def emit(tr, text):
        nonlocal partial_shown, seq
        seq += 1
        raw = s2twp.convert(text.strip())
        line = to_display(text.strip())
        stable, rep, missed_tail = align_final(tr.la_commit, line)
        # 純贅詞碎片(呃/嗯/,然後…)不上畫面、不進 2-pass,只留紀錄
        core = "".join(ch for ch in line if ch.isalnum())
        filler = len(core) <= 6 and all(ch in FILLER_CHARS for ch in core)
        # 跨軌去重:喇叭聲被 mic 收到(實測落後 ~0.14s)→ 兩軌認出同一段話,mic 版標 bleed 丟棄
        bleed = False
        if tr.name == "mic" and lb_tr is not None and not filler:
            import difflib
            nrm = lambda s: "".join(ch for ch in s if ch.isalnum()).lower()
            a = nrm(line)
            cands = [to_display(rec.get_result(lb_tr.stream_s))]
            cands += [t2 for (s0, t2) in lb_tr.recent_finals
                      if s0 >= (tr.utt_start or 0) - 10]
            for c in cands:
                b = nrm(c)
                if a and b and (difflib.SequenceMatcher(None, a, b).ratio() >= 0.55
                                or (len(a) >= 10 and (a in b or b in a))):
                    bleed = True
                    break
        tr.recent_finals.append((tr.utt_start or time.time(), line))
        del tr.recent_finals[:-6]
        if not filler and not bleed:
            ov_send({"kind": "final", "seq": seq, "utt": tr.utt_id,
                     "text": stable, "replace_last": rep})
        else:
            ov_send({"kind": "retract", "utt": tr.utt_id})  # 丟棄句要收回畫面上的進行中行
        rlog({"ev": "final", "utt": tr.utt_id, "seq": seq, "track": tr.name,
              "audio_start": round((tr.utt_start_sample or 0) / tr.sr, 2),
              "audio_end": round(tr.total / tr.sr, 2),
              "dropped": "filler" if filler else ("bleed" if bleed else None),
              "missed_tail": missed_tail, "tail_replaced": rep,
              "revised_vs_shown": stable != line})
        ts = dt.datetime.fromtimestamp(tr.utt_start or time.time())
        sys.stdout.write("\r" + " " * len(partial_shown.encode("gbk", "replace")) + "\r")
        partial_shown = ""
        tag = "(贅詞略)" if filler else ("(回授略)" if bleed else "")
        print(f"[{ts:%H:%M:%S}][{tr.label}]{tag} {line}", flush=True)
        rec_line = {"t": ts.isoformat(timespec="seconds"),
                    "start": round(tr.utt_start or time.time(), 2),
                    "track": tr.name, "label": tr.label, "text": line}
        if line != raw:
            rec_line["raw"] = raw
        if filler:
            rec_line["filler"] = True
        if bleed:
            rec_line["bleed"] = True
        if stable != line:
            rec_line["shown"] = stable
        if missed_tail:
            rec_line["missed_tail"] = True
        jsonl.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
        jsonl.flush()
        records.append(rec_line)
        if refine_dir is not None and tr.utt_start_sample is not None and not filler and not bleed:
            clip = tr.slice_from(tr.utt_start_sample - int(1.5 * tr.sr))  # 前補 1.5s 防切頭
            qw = wave.open(str(refine_dir / f"{seq:06d}.wav"), "wb")
            qw.setnchannels(1)
            qw.setsampwidth(2)
            qw.setframerate(tr.sr)
            qw.writeframes(clip.tobytes())
            qw.close()
            (refine_dir / f"{seq:06d}.json").write_text(json.dumps(
                {"seq": seq, "t": rec_line["t"], "track": tr.name, "wall": time.time(),
                 "utt": tr.utt_id, "label": tr.label, "draft": line},
                ensure_ascii=False), encoding="utf-8")

    ov_live_sent = [None]
    display_owner = [None]
    last_busy = [0.0]
    utt_counter = [0]
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
                    tr.utt_start_sample = tr.total
                    utt_counter[0] += 1
                    tr.utt_id = utt_counter[0]
                    tr.la_last = ""
                    tr.la_commit = ""
                    tr.la_t = tr.utt_start
                    tr.suspect = False
                if rec.is_endpoint(tr.stream_s):
                    if text.strip():
                        emit(tr, text)
                    rec.reset(tr.stream_s)
                    tr.utt_start = None
                    tr.utt_start_sample = None
                    tr.utt_id = None
                    continue
            while not refined_q.empty():
                ln = refined_q.get()
                sys.stdout.write("\r" + " " * len(partial_shown.encode("gbk", "replace")) + "\r")
                partial_shown = ""
                print(ln, flush=True)
            # LA 時鐘每軌獨立跑:不論誰在畫面上,各軌的提交前綴照常累積
            talking = [tr for tr in tracks if rec.get_result(tr.stream_s)]
            now = time.time()
            for tr in talking:
                full = to_display(rec.get_result(tr.stream_s))
                if now - tr.la_t >= 0.4:  # 取樣節奏:兩輪一致的前綴才提交
                    tr.la_t = now
                    if tr.la_last:
                        n = 0
                        for a, b in zip(tr.la_last, full):
                            if a != b:
                                break
                            n += 1
                        cand = full[:n]
                        # 提交點不落在英文單字中間(避免 mic wa 這種半個單字上畫面)
                        while (cand and cand[-1].isascii() and cand[-1].isalpha()
                               and n < len(full) and full[n].isascii() and full[n].isalpha()):
                            cand = cand[:-1]
                            n -= 1
                        # 孤立單字不單獨提交:一次至少推進 2 字,等下一批一起顯示
                        if len(cand) - len(tr.la_commit) >= 2 and cand.startswith(tr.la_commit):
                            tr.la_commit = cand
                            rlog({"ev": "live_commit", "utt": tr.utt_id, "track": tr.name,
                                  "len": len(cand)})
                    # 浮動尾端上限 12 字:更早的內容即使未達成共識也強制提交
                    if full.startswith(tr.la_commit) and len(full) - len(tr.la_commit) > 12:
                        tr.la_commit = full[:len(full) - 12]
                    tr.la_last = full
                    # 回音提前攔截:mic 進行中文字與 loopback 相似 → 本句標 suspect,不得上畫面
                    if (tr.name == "mic" and not tr.suspect and lb_tr is not None
                            and len(full) >= 6 and now - lb_tr.last_loud < 2.0):
                        import difflib
                        b = to_display(rec.get_result(lb_tr.stream_s))
                        na = "".join(c for c in full if c.isalnum()).lower()
                        nb = "".join(c for c in b if c.isalnum()).lower()
                        if na and nb and difflib.SequenceMatcher(None, na, nb).ratio() > 0.6:
                            tr.suspect = True
                            rlog({"ev": "suspect", "utt": tr.utt_id})
            # 活動心跳:任一軌句子進行中就禁止 overlay 清屏(清除倒數只從 final 起算)
            if any(t2.utt_id is not None for t2 in tracks) and now - last_busy[0] > 1.0:
                ov_send({"kind": "busy"})
                last_busy[0] = now
            # 顯示仲裁:①喇叭出聲期間 mic 多半是回授,不得搶畫面;
            # ②一句講完前畫面不換軌(所有權),另一軌內容等定稿再以 final 併入
            cands = [t2 for t2 in talking
                     if not t2.suspect
                     and not (t2.name == "mic" and lb_tr is not None
                              and now - lb_tr.last_loud < 0.7)]
            owner = display_owner[0]
            if owner not in cands or owner.utt_id is None:
                owner = min(cands, key=lambda t2: t2.utt_start or now) if cands else None
                display_owner[0] = owner
            show = ""
            if owner is not None:
                tr = owner
                full = to_display(rec.get_result(tr.stream_s))
                tail = full[len(tr.la_commit):] if full.startswith(tr.la_commit) else ""
                show = f"… [{tr.label}] {tr.la_commit + tail}"[-80:]
                st = (tr.utt_id, tr.la_commit, tail)
                if st != ov_live_sent[0]:
                    ov_send({"kind": "live", "utt": tr.utt_id,
                             "committed": tr.la_commit, "tail": tail})
                    ov_live_sent[0] = st
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
        if refiner_proc is not None:
            (refine_dir / "DONE").touch()
            print("\n等待 2-pass 修正收尾 ...", flush=True)
            deadline = time.time() + 300
            while (refiner_proc.poll() is None or not refined_q.empty()) and time.time() < deadline:
                try:
                    print(refined_q.get(timeout=0.3), flush=True)
                except queue.Empty:
                    pass
            if refiner_proc.poll() is None:
                refiner_proc.terminate()
        if overlay_proc is not None:
            try:
                overlay_proc.stdin.close()  # overlay 收到 EOF 自行關閉
            except OSError:
                pass
            try:
                overlay_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                overlay_proc.terminate()
        # 依 audio_start 排序的整併版(兩軌同時完成時 subtitles.jsonl 是寫入序,會倒置)
        with open(session / "subtitles_sorted.jsonl", "w", encoding="utf-8") as sf:
            for r in sorted(records, key=lambda r: r.get("start", 0)):
                sf.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n--- 結束,產物在 {session} ---")
        for tr in tracks:
            tr.close()
        jsonl.close()
        render_log.close()
        p.terminate()


if __name__ == "__main__":
    main()
