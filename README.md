# VoiceSTT — 會議錄音 → 逐字稿/摘要/待辦(全本機、Claude Code skill)

把會議或演講錄音(mp3 / m4a / wav / mp4)變成:**帶說話人與時間戳的逐字稿**、**會議摘要**、**待辦清單**。針對台灣華語與中英夾雜最佳化,**音訊全程本機處理、絕不上傳任何服務**——只有轉出的文字會進入 AI 對話。

設計上是一個 [Claude Code](https://claude.com/claude-code) skill:裝好之後,對 Claude 丟一句「`<音檔> 產逐字稿和會議紀錄`」,它會照本 repo 固化的 SOP 全自動跑完管線、裁決說話人數、寫出兩份交付檔並自我查核。

## 快速開始(Claude CLI,一鍵)

前置:Windows 10/11、PowerShell 7、Python 3.10+(`py` launcher)、已安裝 Claude Code。NVIDIA GPU 可選(有則自動加速 3–4 倍)。

```powershell
git clone https://github.com/jonny50905/VoiceSTT.git
cd VoiceSTT
.\install.ps1        # 裝 skill + 建 venv + 下載模型(~500MB,冪等)
```

然後在任何資料夾:

```powershell
claude "D:\rec\meeting.m4a 產逐字稿和會議紀錄"
```

Claude 會自動觸發 `meeting-minutes` skill,依 SKILL.md 的 SOP:背景跑管線 → 讀診斷表裁決說話人數 → 產出 `<音檔名>_逐字稿.md` 與 `<音檔名>_會議摘要與待辦.md` 到音檔同目錄,並對摘要的數字與歸屬做查核。

## 不經 Claude 的直跑(保留彈性)

只要轉錄結果、不需要 AI 整理時:

```powershell
.\voicestt.ps1 "D:\rec\meeting.m4a"                      # 全管線
.\voicestt.ps1 "D:\rec\talk.m4a" --stage merge --speakers 1   # 改判單人後重併
```

產物在 `%LOCALAPPDATA%\meeting-minutes\work\<音檔名>\`,主要是 `transcript_body.md`(逐字稿正文)與 `recluster_*.json`(說話人數診斷)。

## 管線架構與模型

| 步驟 | 做什麼 | 模型 | 裝置 |
|---|---|---|---|
| decode | 音檔 → 16kHz PCM | (PyAV,免裝 ffmpeg) | CPU |
| diarize | 語音切分成 turns | pyannote segmentation-3.0(sherpa-onnx) | GPU/CPU |
| transcribe | 語音辨識(zh-TW/中英夾雜) | Breeze-ASR-25(聯發科,faster-whisper CT2) | GPU/CPU |
| recluster | 每個 turn 抽聲紋、聚類、選說話人數 | 3D-Speaker ERes2Net(主)+ CAM++(交叉驗證)+ 自製 AHC | GPU/CPU |
| merge | 逐字歸屬說話人、併句、簡繁/標點後處理 | OpenCC s2twp + CT-Transformer 標點 | CPU |

- **斷點續跑**:各 stage 有產物即跳過;diarize 內部再以 10 分鐘為一塊落盤 checkpoint(綁定音訊 SHA-1),意外中斷重跑同指令只損失當塊。
- **效能**(RTX 4070 SUPER 實測):整條約 **0.55× 音檔長度**(兩小時會議約 1 小時);無 GPU 約 1.7×。
- 選型為 2026-07 查證定案(Breeze-ASR-25 在台灣華語勝過 whisper-large-v3);時隔一年以上建議先重新查證。

## 說話人數裁決(SOP 核心,勿盲信自動值)

自動選的 k 只是初值。SKILL.md 內建裁決準則:使用者知道人數以其為準;次群佔比過低判單人;聲音相近被併群時用 `inspect_split.py` 看三項證據(子聚類時長、跨模型交叉表、中位 F0)再拆。改判後只需重跑 merge(秒級)。

進階場景(SKILL.md 有完整方法):

- **劣化音訊**(線上會議混音檔):先要平台原生逐字稿 > 換模型族 > 標記遺失區段;音訊增強實測全部無效,別浪費時間。
- **同場多音源**:遠場音源聲紋塌縮時,可從另一音源的 diarization 跨錄音移植說話人標籤(能量互相關+文字錨點對齊,聲紋質心驗證)。

## 目錄結構

```
SKILL.md              # Claude Code skill 主檔:SOP、裁決準則、地雷表(單一權威來源)
scripts/
  setup.ps1           # 冪等環境建置:venv、依賴釘版、模型下載、GPU 自動偵測
  pipeline.py         # 五階段管線,單檔零依賴框架
  inspect_split.py    # 聚類法醫:併群檢查與手動拆分
install.ps1           # 一鍵裝進 ~\.claude\skills\meeting-minutes
voicestt.ps1          # 不經 Claude 的直跑入口
```

執行環境(venv + 模型)統一放 `%LOCALAPPDATA%\meeting-minutes\`,與 repo 及各專案目錄脫鉤;更新 skill 重跑 `install.ps1` 即可。

## 已知地雷(完整清單見 SKILL.md,已寫死在腳本裡勿改)

- 聲紋模型必用 `embed_eres2net_common`;sherpa 官方範例那顆小語料模型會把 4 人切成 70 人。
- `provider="cuda"` 在 CPU wheel 上會**靜默 fallback 不報錯**;diarize 異常慢先用 `nvidia-smi` 看 VRAM。
- 同時只跑一個 GPU diarize 程序;遇 VRAM 問題縮 `DIAR_BLOCK`,別把 embedding 改回 CPU(慢 4 倍)。
- transcribe 的抗幻覺參數組是社群共識,勿「順手優化」。
- Windows:HF cache 用預設路徑(260 字元上限)、CUDA 靠 pip 版 DLL + `add_dll_directory`。

## 隱私

音訊檔全程本機處理,不上傳任何服務;模型皆為本機推論。經 Claude 使用時,只有**轉出的文字**會進入對話(等同你貼文字給 Claude)。逐字稿與會議內容屬敏感資料,請勿放進版本控制——本 repo 的 `.gitignore` 已排除常見音檔與產物格式。
