"""會議錄音 → 說話人分離逐字稿管線(音訊全程本機處理,不上傳)。

用法:
  python pipeline.py <音檔> [--stage all|decode|diarize|transcribe|recluster|merge]
                     [--speakers N] [--labels 檔名] [--embedding 名稱] [--force]

產物在 %LOCALAPPDATA%\\meeting-minutes\\work\\<音檔名>\\,各 stage 已有產物即跳過(--force 重跑)。
模型選型與參數為 2026-07 上網查證後定案;修改前先讀 SKILL.md 的地雷表。
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Windows 無系統 CUDA 時,把 pip 版 CUDA DLL(cublas/cudnn/cudart/cufft)加進搜尋路徑;
# sherpa-onnx CUDA wheel 與 ctranslate2 都靠這批 DLL,須在 import 前掛好
_nvidia_base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
for _b in sorted(_nvidia_base.glob("*/bin")):
    os.add_dll_directory(str(_b))
    os.environ["PATH"] = str(_b) + os.pathsep + os.environ["PATH"]

ROOT = Path(os.environ["LOCALAPPDATA"]) / "meeting-minutes"
MODELS = ROOT / "models"
SEG_MODEL = str(MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx")
PUNCT_MODEL = str(MODELS / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12" / "model.onnx")
ASR_REPO = "SoybeanMilk/faster-whisper-Breeze-ASR-25"  # Breeze-ASR-25 社群 CT2 轉檔
SR = 16000
MAIN_SHARE = 0.02  # merge:佔比 < 2% 的聚類視為雜訊,其發言由前後文歸屬


def wp(args, name):
    return args.workdir / name


def stage_decode(args):
    out = wp(args, "audio_16k.npy")
    if out.exists() and not args.force:
        print("decode: 已有 audio_16k.npy,跳過")
        return
    import numpy as np
    from faster_whisper import decode_audio  # 內建 PyAV 解碼 mp3/m4a/wav,免裝 ffmpeg

    t0 = time.time()
    audio = np.asarray(decode_audio(str(args.audio), sampling_rate=SR), dtype=np.float32)
    np.save(out, audio)
    d = len(audio) / SR
    print(f"decode: {d:.1f}s ({int(d // 60)}m{int(d % 60):02d}s),{time.time() - t0:.0f}s")


DIAR_BLOCK = 600.0  # 分塊斷點續跑:每塊 10 分鐘,意外中斷只重跑當塊
DIAR_OVL = 30.0     # 塊間重疊;權責邊界取重疊中點,±15s 保證接縫幀仍有完整滑窗上下文


def stage_diarize(args):
    out = wp(args, "diar_turns.json")
    if out.exists() and not args.force:
        print("diarize: 已有 diar_turns.json,跳過")
        return
    import numpy as np
    import sherpa_onnx

    audio = np.load(wp(args, "audio_16k.npy"))
    total = len(audio) / SR
    step = DIAR_BLOCK - DIAR_OVL
    n_blocks = max(1, math.ceil((total - DIAR_OVL) / step))
    # 塊參數寫進目錄名:改常數後舊 checkpoint 自然失效,不會被誤拼
    blkdir = wp(args, f"diar_blocks_{int(DIAR_BLOCK)}_{int(DIAR_OVL)}")
    blkdir.mkdir(exist_ok=True)

    # checkpoint 綁定音訊內容:同名檔換內容(--force decode)後,舊 block 必須作廢,
    # 否則會被靜默拼出「合法但錯位」的 turns
    import hashlib
    h = hashlib.sha1()
    with open(wp(args, "audio_16k.npy"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    meta = {"audio_sha1": h.hexdigest()}
    metaf = blkdir / "meta.json"
    stale = True
    if metaf.exists():
        try:
            with open(metaf, encoding="utf-8") as f:
                stale = json.load(f) != meta
        except Exception:
            stale = True
    if stale:
        for p in blkdir.glob("block_*.json"):
            p.unlink()
        tmp = metaf.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, metaf)

    # provider="cuda" 需要 CUDA wheel(sherpa-onnx==x.y.z+cuda12.cudnn9);
    # CPU wheel 會「靜默」fallback CPU 不報錯,速度異常時先懷疑這裡
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=8, provider="cuda",
        ),
        # embedding 也走 GPU:sherpa 內部對「每個滑窗」抽聲紋,放 CPU 會慢 4 倍(實測)。
        # 單體 2h 檔曾因此 MemoryError,但分塊後每次 process 只有 600s 量級,VRAM 有界;
        # 若再遇 MemoryError,先縮 DIAR_BLOCK,不要把這裡改回 CPU
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(MODELS / "embed_eres2net_common.onnx"), num_threads=8, provider="cuda"
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
        min_duration_on=0.2,
        min_duration_off=0.5,
    )
    sd = None
    t0 = time.time()
    block_secs = []  # 已完成塊的耗時,滾動估 ETA
    for i in range(n_blocks):
        bf = blkdir / f"block_{i:04d}.json"
        if bf.exists() and not args.force:
            continue
        if sd is None:
            sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        s = i * step
        e = total if i == n_blocks - 1 else min(total, s + DIAR_BLOCK)
        tb = time.time()
        last_beat = [tb]

        def progress(processed, tot, _i=i, _last=last_beat):
            # 主進度以「塊完成」為單位(見下方 ETA 行);這裡只做塊內心跳,
            # 用時間節流(≥60s 一行):GPU 塊短於 60s 幾乎不印,CPU 一塊 15 分鐘
            # 也能看出還活著,不會回到每窗一行的進度洪水
            now = time.time()
            if now - _last[0] >= 60 and processed < tot:
                _last[0] = now
                print(f"diarize: block {_i + 1}/{n_blocks} {processed / tot:.0%}", flush=True)
            return 0

        result = sd.process(audio[int(s * SR):int(e * SR)], callback=progress).sort_by_start_time()
        block_secs.append(time.time() - tb)
        eta = (n_blocks - i - 1) * (sum(block_secs) / len(block_secs))
        print(f"diarize: block {i + 1}/{n_blocks} 完成,{block_secs[-1]:.0f}s,預估剩 {eta / 60:.1f}m",
              flush=True)
        # 只取 turn 邊界;此處的 speaker 標籤品質不足採信,recluster stage 會重新聚類
        turns = [{"start": s + float(r.start), "end": s + float(r.end)} for r in result]
        tmp = bf.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(turns, f)
        os.replace(tmp, bf)

    # 拼接:各塊只保留自己權責區間內的部分(邊界=重疊中點,首尾塊到檔案端點)。
    # 跨接縫的 turn 會被切成兩段,不需縫合:recluster 對每個 turn 獨立抽聲紋再全域
    # 聚類,merge 又會把同說話人相鄰片段併回同一句,下游自然吸收。
    all_turns = []
    for i in range(n_blocks):
        with open(blkdir / f"block_{i:04d}.json", encoding="utf-8") as f:
            turns = json.load(f)
        lo = 0.0 if i == 0 else i * step + DIAR_OVL / 2
        hi = total if i == n_blocks - 1 else i * step + DIAR_BLOCK - DIAR_OVL / 2
        for t in turns:
            ts, te = max(t["start"], lo), min(t["end"], hi)
            if te - ts > 0.05:
                all_turns.append({"start": ts, "end": te})
    all_turns.sort(key=lambda t: t["start"])
    tmp = out.with_suffix(".tmp")  # 原子寫入:截斷檔會被 exists() 誤判為「已完成」
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_turns, f)
    os.replace(tmp, out)
    print(f"diarize: {len(all_turns)} turns({n_blocks} blocks),{time.time() - t0:.0f}s")


def stage_transcribe(args):
    out = wp(args, "whisper_segments.json")
    if out.exists() and not args.force:
        print("transcribe: 已有 whisper_segments.json,跳過")
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # 重導向輸出下進度條會印亂碼
    import numpy as np
    from huggingface_hub import snapshot_download
    from faster_whisper import WhisperModel

    # 必須下載到預設 HF cache:自訂深層目錄會爆 Windows 260 字元路徑上限
    model_dir = snapshot_download(ASR_REPO)
    audio = np.load(wp(args, "audio_16k.npy"))
    dur = len(audio) / SR

    def run(device, compute_type):
        t0 = time.time()
        model = WhisperModel(model_dir, device=device, compute_type=compute_type,
                             cpu_threads=10 if device == "cpu" else 0)
        print(f"transcribe: model on {device} ({time.time() - t0:.0f}s)", flush=True)
        # 抗幻覺參數組:2026-07 社群對中文長音檔的共識(faster-whisper discussion #349 等)
        segments, _ = model.transcribe(
            audio,
            language="zh",
            task="transcribe",
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            compression_ratio_threshold=None,
            log_prob_threshold=None,
            no_speech_threshold=None,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
        )
        res = []
        for i, s in enumerate(segments):
            res.append({
                "start": s.start, "end": s.end, "text": s.text,
                "words": [{"start": w.start, "end": w.end, "word": w.word, "p": w.probability}
                          for w in (s.words or [])],
            })
            if i % 25 == 0:
                print(f"transcribe: {s.end:.0f}/{dur:.0f}s", flush=True)
        print(f"transcribe: {len(res)} segments,{time.time() - t0:.0f}s on {device}")
        return res

    try:
        res = run("cuda", "float16")
    except Exception as e:
        print(f"transcribe: GPU 失敗({e!r}),改用 CPU int8", flush=True)
        res = run("cpu", "int8")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


def stage_recluster(args):
    out = wp(args, f"recluster_{args.embedding}.json")
    embf = wp(args, f"emb_{args.embedding}.npy")
    if out.exists() and not args.force:
        print(f"recluster: 已有 {out.name},跳過")
        return
    import numpy as np
    import sherpa_onnx
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_samples

    audio = np.load(wp(args, "audio_16k.npy"))
    with open(wp(args, "diar_turns.json"), encoding="utf-8") as f:
        turns = json.load(f)

    if embf.exists() and not args.force:
        embs = np.load(embf)
    else:
        # 同一程序先跑過 transcribe(ctranslate2)後,ORT 的 CUDA provider 會載入失敗
        # (Error 1114);fresh process 則正常。失敗就退 CPU,勿讓整個 stage 死掉
        try:
            ext = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(MODELS / f"{args.embedding}.onnx"), num_threads=8, provider="cuda")
            )
        except RuntimeError as e:
            print(f"recluster: CUDA embedding 失敗({e!r}),改用 CPU", flush=True)
            ext = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(MODELS / f"{args.embedding}.onnx"), num_threads=8)
            )
        embs = []
        for i, t in enumerate(turns):
            s, e = int(t["start"] * SR), int(t["end"] * SR)
            if e - s < SR:  # 過短片段聲紋不可靠,前後各補 0.3s context
                s, e = max(0, s - int(0.3 * SR)), min(len(audio), e + int(0.3 * SR))
            st = ext.create_stream()
            st.accept_waveform(SR, audio[s:e])
            st.input_finished()
            v = np.array(ext.compute(st), dtype=np.float32)
            embs.append(v / (np.linalg.norm(v) + 1e-9))
            if i % 50 == 0:
                print(f"recluster: embed {i}/{len(turns)}", flush=True)
        embs = np.stack(embs)
        np.save(embf, embs)

    durs = np.array([t["end"] - t["start"] for t in turns])
    res = {"embedding": args.embedding, "results": {}, "labels": {}}
    kmax = min(8, len(turns) - 1)
    for k in range(2, kmax + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(embs)
        sil = silhouette_samples(embs, lab, metric="cosine")
        wsil = float(np.average(sil, weights=durs))  # 長發言權重高
        shares = sorted((float(durs[lab == i].sum() / durs.sum()) for i in range(k)), reverse=True)
        res["results"][str(k)] = {"weighted_sil": wsil, "shares": shares}
        res["labels"][str(k)] = lab.tolist()
        print(f"recluster: k={k} wsil={wsil:.3f} shares={[f'{s:.0%}' for s in shares]}")
    best = max(res["results"], key=lambda k: res["results"][k]["weighted_sil"])
    res["best_k"] = int(best)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f)
    print(f"recluster: best_k={best}(初值,說話人數仍需人工裁決,見 SKILL.md)")
    if res["results"]["2"]["shares"][1] < 0.03:
        print("recluster hint: k=2 的次群佔比 <3%,可能為單人錄音 → --stage merge --speakers 1")


def stage_merge(args):
    with open(wp(args, "whisper_segments.json"), encoding="utf-8") as f:
        segments = json.load(f)
    words = []
    for s in segments:
        if s["words"]:
            words.extend(s["words"])
        elif s["text"].strip():
            words.append({"start": s["start"], "end": s["end"], "word": s["text"]})
    words.sort(key=lambda w: w["start"])

    if args.speakers == 1:
        for w in words:
            w["spk"] = 0
    else:
        with open(wp(args, "diar_turns.json"), encoding="utf-8") as f:
            turns = json.load(f)
        if args.labels:
            with open(wp(args, args.labels), encoding="utf-8") as f:
                data = json.load(f)
            labels = data["labels"] if isinstance(data, dict) else data
        else:
            with open(wp(args, f"recluster_{args.embedding}.json"), encoding="utf-8") as f:
                rc = json.load(f)
            k = str(args.speakers or rc["best_k"])
            if k not in rc["labels"]:
                sys.exit(f"merge: recluster 結果沒有 k={k};可用:{sorted(rc['labels'])}")
            labels = rc["labels"][k]
        if len(labels) != len(turns):
            sys.exit(f"merge: labels 數({len(labels)})≠ turns 數({len(turns)})")

        durs, total = {}, 0.0
        for t, l in zip(turns, labels):
            d = t["end"] - t["start"]
            durs[l] = durs.get(l, 0) + d
            total += d
        main = {l for l, d in durs.items() if d / total >= MAIN_SHARE}
        print(f"merge: main clusters {sorted(main)}(全部 {len(durs)} 群)")
        turns = [{**t, "speaker": l} for t, l in zip(turns, labels) if l in main]

        def overlap_speaker(ws, we):
            best, best_ov = None, 0.0
            for t in turns:
                ov = min(we, t["end"]) - max(ws, t["start"])
                if ov > best_ov:
                    best_ov, best = ov, t["speaker"]
            return best

        def nearest_speaker(mid, max_dist=2.0):
            best, best_d = None, max_dist
            for t in turns:
                d = max(t["start"] - mid, mid - t["end"], 0.0)
                if d < best_d:
                    best_d, best = d, t["speaker"]
            return best

        for w in words:
            spk = overlap_speaker(w["start"], w["end"])
            if spk is None:
                spk = nearest_speaker((w["start"] + w["end"]) / 2)
            w["spk"] = spk
        prev = None
        for w in words:  # 落單的字先繼承前一位、再繼承後一位說話人
            if w["spk"] is None:
                w["spk"] = prev
            prev = w["spk"]
        nxt = None
        for w in reversed(words):
            if w["spk"] is None:
                w["spk"] = nxt
            nxt = w["spk"]

    utts = []
    cur = None
    for w in words:
        long_break = cur and cur["end"] - cur["start"] > 60.0 and w["start"] - cur["end"] > 0.6
        if cur is None or w["spk"] != cur["spk"] or w["start"] - cur["end"] > 4.0 or long_break:
            if cur:
                utts.append(cur)
            cur = {"spk": w["spk"], "start": w["start"], "end": w["end"], "text": w["word"]}
        else:
            cur["text"] += w["word"]
            cur["end"] = w["end"]
    if cur:
        utts.append(cur)

    from opencc import OpenCC
    import sherpa_onnx

    cc = OpenCC("s2twp")  # Breeze 輸出偶漏簡體字,統一轉台灣正體
    punct = sherpa_onnx.OfflinePunctuation(
        sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(ct_transformer=PUNCT_MODEL, num_threads=4)
        )
    )

    def polish(text):
        text = cc.convert(text.strip())
        # API 是 add_punctuation(add_punct 不存在);長文切 400 字分段餵
        parts = [text[i:i + 400] for i in range(0, len(text), 400)]
        text = "".join(punct.add_punctuation(p) for p in parts)
        # 標點模型會吃掉中英之間的空格,補回
        text = re.sub(r"([一-鿿])([A-Za-z0-9])", r"\1 \2", text)
        text = re.sub(r"([A-Za-z0-9])([一-鿿])", r"\1 \2", text)
        return text

    order = []
    for u in utts:
        if u["spk"] not in order:
            order.append(u["spk"])
    name = {spk: f"發言人{chr(ord('A') + i)}" for i, spk in enumerate(order)}

    def ts(sec):
        sec = int(sec)
        return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"

    lines, talk_time, out_utts = [], {}, []
    for u in utts:
        text = polish(u["text"])
        if not text:
            continue
        n = name[u["spk"]]
        lines.append(f"**[{ts(u['start'])} - {ts(u['end'])}] {n}:** {text}")
        talk_time[n] = talk_time.get(n, 0) + (u["end"] - u["start"])
        out_utts.append({"name": n, "start": u["start"], "end": u["end"], "text": text})

    with open(wp(args, "transcript_body.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    with open(wp(args, "utts.json"), "w", encoding="utf-8") as f:
        json.dump(out_utts, f, ensure_ascii=False, indent=1)
    print(f"merge: {len(out_utts)} utterances → transcript_body.md")
    for n in sorted(talk_time):
        print(f"  {n}: {talk_time[n] / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--stage", default="all",
                    choices=["all", "decode", "diarize", "transcribe", "recluster", "merge"])
    ap.add_argument("--speakers", type=int, help="指定說話人數(1=單人錄音,跳過分離)")
    ap.add_argument("--labels", help="workdir 內的自訂標籤檔(inspect_split.py --split 產出)")
    ap.add_argument("--embedding", default="embed_eres2net_common",
                    help="聲紋模型(models/ 下檔名去掉 .onnx;交叉驗證用 embed_campplus_zhen)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.audio = args.audio.resolve()
    if not args.audio.exists():
        sys.exit(f"找不到音檔:{args.audio}")
    args.workdir = ROOT / "work" / args.audio.stem
    args.workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {args.workdir}")
    stages = ["decode", "diarize", "transcribe", "recluster", "merge"] if args.stage == "all" else [args.stage]
    for s in stages:
        globals()[f"stage_{s}"](args)
    print("PIPELINE DONE")


if __name__ == "__main__":
    main()
