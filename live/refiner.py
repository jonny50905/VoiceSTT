# -*- coding: utf-8 -*-
"""句尾 2-pass 修正程序(Phase 2):Breeze-ASR-25 重跑每句定稿音訊,出高品質修正行。

由 live_subtitles.py 以子程序啟動(手動偵錯亦可單獨跑):
  python refiner.py <session_dir> [--terms terms.txt] [--device cuda|cpu]

- 讀 <session_dir>/refine_queue/NNNNNN.json(+同名 .wav),依序重跑
- Breeze(faster-whisper CT2,GPU)+ CT-Transformer 標點(CPU)+ s2twp + 術語映射
- 修正行寫 <session_dir>/subtitles_refined.jsonl 並印到 stdout(⟳ 前綴,母程序轉印)
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

from live_subtitles import BASE, load_terms  # 亦順帶掛好 CUDA DLL 路徑

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
        print(f"⟳ Breeze {args.device} 失敗({e}),退回 CPU", flush=True)
        model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=10)
    punct = sherpa_onnx.OfflinePunctuation(
        sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=PUNCT_MODEL, num_threads=4)
        )
    )
    cc = OpenCC("s2twp")
    terms = load_terms(Path(args.terms)) if args.terms else []
    print(f"⟳ 2-pass 就緒({time.time() - t0:.0f}s)", flush=True)

    out = open(session / "subtitles_refined.jsonl", "a", encoding="utf-8")
    while True:
        items = sorted(qdir.glob("*.json"))
        if not items:
            if (qdir / "DONE").exists():
                break
            time.sleep(0.3)
            continue
        for jp in items:
            meta = json.loads(jp.read_text(encoding="utf-8"))
            wp = jp.with_suffix(".wav")
            w = wave.open(str(wp))
            sr = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            w.close()
            audio = pcm.astype(np.float32) / 32768.0
            if sr != 16000:  # faster-whisper 吃 16k;整數比率線性內插即可
                idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
                audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
            # 抗幻覺參數組同 pipeline.py(短句免 VAD/斷句相關項)
            segments, _ = model.transcribe(
                audio, language="zh", task="transcribe",
                beam_size=5, best_of=5, condition_on_previous_text=False,
                compression_ratio_threshold=None, log_prob_threshold=None,
                no_speech_threshold=None, repetition_penalty=1.05, no_repeat_ngram_size=3,
            )
            text = "".join(s.text for s in segments).strip()
            if text:
                text = cc.convert(punct.add_punctuation(text))
                for a, b in terms:
                    text = text.replace(a, b)
            rec_line = {**meta, "text": text or meta["draft"], "kept_draft": not text}
            if text:
                del rec_line["kept_draft"]
            out.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
            out.flush()
            print(f"⟳[{meta['t'][11:]}][{meta['label']}] {rec_line['text']}", flush=True)
            wp.unlink()
            jp.unlink()
    out.close()


if __name__ == "__main__":
    main()
