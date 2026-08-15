"""聚類法醫:檢查某聚類是否為兩位聲音相近的說話人被併群(前例:一女三男被併成 3 人)。

用法:
  python inspect_split.py <音檔> --cluster N [--k K] [--embedding embed_eres2net_common]
                          [--alt embed_campplus_zhen] [--split]

輸出三項證據:
  1. 子聚類二分後各群發言時長
  2. 與另一顆聲紋模型分群結果的交叉表(需先跑 pipeline.py --stage recluster --embedding <alt>)
  3. 各子群中位 F0 —— 女聲約 165–255Hz,男聲約 85–155Hz
三項都指向兩人,才視為真實拆分。--split 會把該聚類拆成兩群寫入 recluster_custom.json,
之後用 pipeline.py <音檔> --stage merge --labels recluster_custom.json 重產逐字稿。
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(os.environ["LOCALAPPDATA"]) / "meeting-minutes"
SR = 16000


def median_f0(audio, turns, idxs):
    """自相關法逐框抓 F0(640 樣本框、hop 320),回傳中位數與有效框數。"""
    import numpy as np

    f0s = []
    lo, hi = int(SR / 350), int(SR / 60)  # 60–350Hz 的 lag 範圍
    for i in idxs:
        seg = audio[int(turns[i]["start"] * SR):int(turns[i]["end"] * SR)]
        for off in range(0, len(seg) - 640, 320):
            fr = seg[off:off + 640].astype(np.float64)
            fr -= fr.mean()
            ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
            if ac[0] <= 0:
                continue
            lag = lo + int(np.argmax(ac[lo:hi]))
            if ac[lag] > 0.45 * ac[0]:  # 峰值不夠高視為無聲/氣音框,略過
                f0s.append(SR / lag)
    return (float(np.median(f0s)) if f0s else None), len(f0s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--cluster", type=int, required=True, help="要檢查的聚類編號(recluster labels 值)")
    ap.add_argument("--k", type=int, help="用哪個 k 的分群(預設 best_k)")
    ap.add_argument("--embedding", default="embed_eres2net_common")
    ap.add_argument("--alt", default="embed_campplus_zhen")
    ap.add_argument("--split", action="store_true", help="拆成兩群寫入 recluster_custom.json")
    args = ap.parse_args()

    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    W = ROOT / "work" / args.audio.resolve().stem
    audio = np.load(W / "audio_16k.npy")
    with open(W / "diar_turns.json", encoding="utf-8") as f:
        turns = json.load(f)
    with open(W / f"recluster_{args.embedding}.json", encoding="utf-8") as f:
        rc = json.load(f)
    embs = np.load(W / f"emb_{args.embedding}.npy")

    k = str(args.k or rc["best_k"])
    labels = rc["labels"][k]
    idxs = [i for i, l in enumerate(labels) if l == args.cluster]
    if len(idxs) < 4:
        sys.exit(f"聚類 {args.cluster} 只有 {len(idxs)} 個 turns,不足以子聚類")
    print(f"聚類 {args.cluster}(k={k}):{len(idxs)} turns")

    sub = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(embs[idxs])
    for s in (0, 1):
        members = [idxs[j] for j in range(len(idxs)) if sub[j] == s]
        dur = sum(turns[i]["end"] - turns[i]["start"] for i in members)
        f0, n = median_f0(audio, turns, members)
        f0s = f"{f0:.0f}Hz({n} 框)" if f0 else "n/a"
        print(f"  子群{s}:{len(members)} turns,{dur / 60:.1f} min,中位 F0 {f0s}")

    alt_file = W / f"recluster_{args.alt}.json"
    if alt_file.exists():
        with open(alt_file, encoding="utf-8") as f:
            alt = json.load(f)
        alt_labels = alt["labels"][str(alt["best_k"])]
        print(f"  交叉表(vs {args.alt} 的 k={alt['best_k']}):")
        for s in (0, 1):
            members = [idxs[j] for j in range(len(idxs)) if sub[j] == s]
            counts = {}
            for i in members:
                counts[alt_labels[i]] = counts.get(alt_labels[i], 0) + 1
            print(f"    子群{s} → {dict(sorted(counts.items()))}")
    else:
        print(f"  (無 {alt_file.name};先跑 pipeline.py --stage recluster --embedding {args.alt} 才有交叉表)")

    if args.split:
        new_id = max(labels) + 1
        out = list(labels)
        for j, i in enumerate(idxs):
            if sub[j] == 1:
                out[i] = new_id
        with open(W / "recluster_custom.json", "w", encoding="utf-8") as f:
            json.dump({"labels": out, "note": f"cluster {args.cluster} (k={k}) 拆出子群1 → {new_id}"}, f)
        print(f"已寫入 recluster_custom.json(子群1 → 新聚類 {new_id});"
              f"重產:pipeline.py <音檔> --stage merge --labels recluster_custom.json")


if __name__ == "__main__":
    main()
