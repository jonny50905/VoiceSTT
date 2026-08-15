# -*- coding: utf-8 -*-
"""句尾 2-pass 修正程序(Phase 2):Breeze-ASR-25 重跑每句定稿音訊,出高品質修正行。

由 live_subtitles.py 以子程序啟動(手動偵錯亦可單獨跑):
  python refiner.py <session_dir> [--terms terms.txt] [--device cuda|cpu]

- 讀 <session_dir>/refine_queue/NNNNNN.json(+同名 .wav),依序重跑
- Breeze(faster-whisper CT2,GPU)+ CT-Transformer 標點(CPU)+ s2twp + 術語映射
- 修正行寫 <session_dir>/subtitles_offline.jsonl 並印到 stdout(⟳ 前綴,母程序轉印)
- 佇列清空且見到 DONE 檔即退出
- 刻意獨立程序:與 sherpa-onnx CUDA(live 程序)隔離;本程序內 CT2 之後 ORT 只能 CPU(Error 1114),標點模型本來就跑 CPU,安全
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

from live_subtitles import BASE, apply_terms, load_terms  # 亦順帶掛好 CUDA DLL 路徑

ASR_REPO = "SoybeanMilk/faster-whisper-Breeze-ASR-25"  # 同 pipeline.py
PUNCT_MODEL = str(BASE / "models" / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12" / "model.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--terms")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    session = Path(args.session_dir)
    qdir = session / "refine_queue"

    import os
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    from opencc import OpenCC
    import sherpa_onnx

    t0 = time.time()
    model_dir = snapshot_download(ASR_REPO)
    try:
        model = WhisperModel(model_dir, device=args.device,
                             compute_type="float16" if args.device == "cuda" else "int8")
    except Exception as e:
        print(json.dumps({"kind": "status", "msg": f"⟳ Breeze {args.device} 失敗({e}),退回 CPU"}, ensure_ascii=False), flush=True)
        model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=10)
    punct = sherpa_onnx.OfflinePunctuation(
        sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=PUNCT_MODEL, num_threads=4)
        )
    )
    cc = OpenCC("s2twp")
    terms = load_terms(Path(args.terms)) if args.terms else []
    print(json.dumps({"kind": "status", "msg": f"⟳ 2-pass 就緒({time.time() - t0:.0f}s)"}, ensure_ascii=False), flush=True)

    def polish(text):
        return apply_terms(cc.convert(punct.add_punctuation(text)), terms)

    def read_audio(wp):
        w = wave.open(str(wp))
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        w.close()
        audio = pcm.astype(np.float32) / 32768.0
        if sr != 16000:  # faster-whisper 吃 16k;整數比率線性內插即可
            idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        return audio

    def transcribe(audio):
        # 抗幻覺參數組同 pipeline.py(短句免 VAD/斷句相關項)
        segments, _ = model.transcribe(
            audio, language="zh", task="transcribe",
            beam_size=5, best_of=5, condition_on_previous_text=False,
            compression_ratio_threshold=None, log_prob_threshold=None,
            no_speech_threshold=None, repetition_penalty=1.05, no_repeat_ngram_size=3,
        )
        return "".join(s.text for s in segments).strip()

    done_utts = set()
    part_state = {}  # utt -> {"prev": 上次整句解碼, "shown": 已確認前綴, "cut_rev": 前綴對應的快照版}
    out = open(session / "subtitles_offline.jsonl", "a", encoding="utf-8")
    while True:
        items = sorted(qdir.glob("*.json"))
        if not items:
            if (qdir / "DONE").exists():
                break
            time.sleep(0.3)
            continue
        for jp in items:
            try:
                meta = json.loads(jp.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            wp = jp.with_suffix(".wav")
            if meta.get("partial"):
                # 增量快照:同句已有更新版、或該句已定稿 → 過期跳過,不浪費 GPU
                utt, rev = meta["utt"], meta["rev"]
                newer = any(int(p.stem.split("_")[1]) > rev
                            for p in qdir.glob(f"p{utt:06d}_*.json"))
                finalized = utt in done_utts or any(
                    json.loads(p.read_text(encoding="utf-8")).get("utt") == utt
                    for p in qdir.glob("*.json") if p.stem.isdigit())
                if not (newer or finalized):
                    audio = read_audio(wp)
                    text = transcribe(audio)
                    # LocalAgreement-2:連續兩次解碼一致的前綴才升白,白字只增不改,
                    # 避免「新一輪整句重解碼把之前對的字改錯」上畫面
                    st = part_state.setdefault(utt, {"prev": None, "shown": "", "cut_rev": None})
                    if text and st["prev"] is not None:
                        n = 0
                        for a, b in zip(st["prev"], text):
                            if a != b:
                                break
                            n += 1
                        lcp = text[:n]
                        if len(lcp) > len(st["shown"]) and lcp.startswith(st["shown"]):
                            st["shown"] = lcp
                            st["cut_rev"] = rev - 1  # 前綴涵蓋範圍 ≈ 上一版快照的音訊
                    st["prev"] = text or st["prev"]
                    # 送出前重查一次過期:解碼期間該句可能已定稿(競態會把死句白字貼回去)
                    if st["shown"] and utt not in done_utts and not any(
                            json.loads(p.read_text(encoding="utf-8")).get("utt") == utt
                            for p in qdir.glob("*.json") if p.stem.isdigit()):
                        print(json.dumps(
                            {"kind": "refined_partial", "utt": utt, "cut_rev": st["cut_rev"],
                             "text": polish(st["shown"]).rstrip("。，,."),
                             "lag_s": round(time.time() - meta["wall"], 2)},
                            ensure_ascii=False), flush=True)
                wp.unlink()
                jp.unlink()
                continue
            done_utts.add(meta.get("utt"))
            part_state.pop(meta.get("utt"), None)
            audio = read_audio(wp)
            # 過短碎片(音訊去掉 1.5s 前補與 ~0.8s 尾靜音後不足 1s,或草稿 ≤3 字)Breeze 會幻覺,保留草稿
            if len(audio) / 16000 - 2.3 < 1.0 or len(meta["draft"].strip("，。 ,.?？!")) <= 3:
                rec_line = {**meta, "text": meta["draft"], "kept_draft": "short",
                            "lag_s": round(time.time() - meta.get("wall", time.time()), 2)}
                out.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
                out.flush()
                wp.unlink()
                jp.unlink()
                continue
            text = transcribe(audio)
            if text:
                text = polish(text)
            # 守門:亂碼、或與草稿差異過大(自由改寫/幻覺)→ 拒絕校正版,保留草稿
            gate = None
            if text:
                if "�" in text:
                    gate = "replacement_char"
                else:
                    import difflib
                    import re
                    n = lambda s: re.sub(r"[\s\W_]+", "", s).lower()
                    a, b = n(meta["draft"]), n(text)
                    if a and difflib.SequenceMatcher(None, a, b).ratio() < 0.3:
                        gate = "too_different"
            if gate:
                rec_line = {**meta, "text": meta["draft"], "kept_draft": gate}
            else:
                rec_line = {**meta, "text": text or meta["draft"], "kept_draft": not text}
                if text:
                    del rec_line["kept_draft"]
            lag = time.time() - meta["wall"] if "wall" in meta else None
            if lag is not None:
                rec_line["lag_s"] = round(lag, 2)
            out.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({"kind": "offline", "seq": meta.get("seq"), "utt": meta.get("utt"),
                              "t": meta["t"], "label": meta["label"], "text": rec_line["text"],
                              "lag_s": rec_line.get("lag_s")}, ensure_ascii=False), flush=True)
            wp.unlink()
            jp.unlink()
    out.close()


if __name__ == "__main__":
    main()
