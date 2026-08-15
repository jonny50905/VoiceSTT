---
name: meeting-minutes
description: Use when the user provides a meeting or speech recording (mp3/m4a/wav, or mp4/video from Teams/Zoom/Webex) and wants a transcript with speaker labels and timestamps (逐字稿), meeting summary (會議摘要), action items (待辦事項), or speaker diarization — especially zh-TW / 中英夾雜 audio that must stay on the local machine.
---

# 會議錄音 → 逐字稿/摘要/待辦(本機管線)

## Overview

已驗證的本機管線:Breeze-ASR-25(聯發科 zh-TW ASR,GPU)+ sherpa-onnx pyannote(說話人分離)+ OpenCC/標點後處理。
**隱私鐵則:音訊檔全程本機處理、絕不上傳任何服務;只有轉出的文字會進入對話。首次為使用者處理時要主動揭露這點。**

## 執行步驟

以下 `python` 一律指 `%LOCALAPPDATA%\meeting-minutes\venv\Scripts\python.exe`;`<SKILL_DIR>` 指本檔所在目錄。**cwd 通常在使用者專案,一律用絕對路徑執行**。

1. `pwsh -File <SKILL_DIR>\scripts\setup.ps1` — 冪等:建 venv、裝依賴、下載模型(~500MB,已存在即跳過)。
2. `python <SKILL_DIR>\scripts\pipeline.py <音檔>` — decode→diarize→transcribe→recluster→merge 一次跑完,產物在 `%LOCALAPPDATA%\meeting-minutes\work\<音檔名>\`。各 stage 有產物即跳過(斷點續跑;`--force` 重跑);diarize 內部再以 10 分鐘為一塊落盤 checkpoint(綁定音訊雜湊),意外中斷後重跑同一指令只損失當塊。
   **耗時(NVIDIA GPU 機器):整條約 0.55× 音檔長度——diarize ~0.43× 實時(GPU,2026-08-15 起)+ transcribe ~0.1×(GPU,失敗自動 fallback CPU int8)。無 GPU 時 diarize 退回 CPU ~1.5× 實時、整條 ~1.7×。務必背景執行,依此估算等待時間**。
3. 讀 recluster 診斷表與 `transcript_body.md`,**裁決說話人數**(下節)。改判後只需重跑 merge(秒級)。
4. 撰寫兩份交付檔到音檔同目錄(下下節)。

## 說話人數裁決(必要的人工判斷)

- pipeline 用時長加權 silhouette 自動選 k——**只是初值,不可盲信**。
- 使用者若知道實際人數,以使用者為準:`pipeline.py <音檔> --stage merge --speakers N`。
- 最大群以外合計佔比 <5%,或 k=2 次群 <3% → 很可能是單人錄音(演講、講稿預演),用 `--speakers 1`。
- **聲音相近的說話人會被併群**(實例:一女三男被最強聲紋模型併成 3 人)。與會人數對不上時:
  `python scripts\inspect_split.py <音檔> --cluster N --split`
  看三項證據:子聚類時長、跨模型交叉表(先跑 `--stage recluster --embedding embed_campplus_zhen`)、中位 F0(女聲約 165–255Hz、男聲 85–155Hz)。證據齊了才拆,然後 `--stage merge --labels recluster_custom.json`。

## 劣化音訊(線上會議錄製、遠場收音)

線上會議的**混音**錄製檔(典型 16kHz 單聲道 55kbps AAC)辨識率會明顯掉一截。2026-08-05 用一支 2 小時 Teams 錄影(RT60≈0.64s)實測 12 種策略,以平台原生逐字稿為參考文本算涵蓋率,結論如下——**依序做,不要跳過第 1 項**:

1. **先去要平台原生逐字稿**(Teams 會議聊天可下載 .vtt/.docx;Zoom 用 AI Companion 版本才有繁中;Webex AI Assistant)。平台轉錄跑在**混音前的各人上行串流**,品質通常勝過你拿混音檔自跑,還免費附說話人姓名。**這一步的效益遠大於後面所有技術手段。**
   - ⏳ 有保留期限:Teams 預設 120 天、Webex 360 天、Google Meet 只有 30 天。**先備份再說。**
   - ⚠️ Google Meet 的 transcripts **不支援中文**(live captions 支援 ≠ transcripts 支援)。
   - 平台逐字稿常把同一會議室的多人**全掛在同一個名字**下(共用裝置收音)。此時**用本管線的 diarization 去切分,文字用平台的**,兩邊各取所長。
2. **換模型族**——實測唯一有效的手段。`Qwen3-ASR-1.7B` 涵蓋率 **0.753** vs Breeze **0.720**(差距是雜訊的 8 倍),在平台逐字稿放棄處仍有內容。代價:**輸出簡體必須過 OpenCC s2twp**;偶發生成迴圈失控(單一 60 秒視窗曾耗 197 秒只吐 5 字),**務必設 `max_new_tokens` 上限**。裝在獨立 venv,別污染本管線環境。
   - 用法:`pip install qwen-asr`;`Qwen3ASRModel.from_pretrained(...)` → `model.transcribe(audio=(ndarray, 16000), language="Chinese")`。
   - 高 CP 值用法:**只拿它補平台逐字稿的空洞**(有語音活動但無文字處),不必全檔重跑。
3. **標記遺失區段**——把「有語音活動但無文字」的時段列成清單附在逐字稿。這比猜測誠實,也比補洞有用。
4. **同場會議有第二音源時,標籤可跨錄音移植**(2026-08-15 實證):遠場音源的聲紋聚類會被通道特徵蓋過而塌縮(整場併成一群),但只要另一音源的 diarization 分群成功,就能移植——(a) 對齊 offset 用能量包絡 FFT 互相關+轉錄文字 n-gram 錨點雙路互驗(單路低相關=搜尋範圍太窄,擴大再試);(b) 把舊音源 utts 的說話人依 offset 映射到新音源 turns;(c) 用移植標籤在新音源聲紋上算時長加權質心、全場重指派,一致率 >90% 即可信,無對應段(對方漏錄/尾段)用質心指派補上。副產品:兩檔起迄差異會揭露誰漏錄了開場/散會段。

### ❌ 實測無效,不要再試(2026-08-05 量化)

| 手段 | 實測結果 |
|---|---|
| 音訊增強 / 降噪(GTCRN、DPDFNet、WPE) | **全部有害或無效**。線上會議錄製的 SNR 通常已達 50dB+,沒有噪音可降,只會削掉弱音節。DPDFNet 把 60% 內容當噪音壓掉 |
| `initial_prompt` / `hotwords` 灌領域詞 | 涵蓋率 0.719 / 0.718 vs base 0.720,**無效**。且 hotwords 有災難性失敗風險(某視窗涵蓋率掉到 0.04,模型抓著提示詞硬湊) |
| 放寬 VAD `min_silence_duration_ms` | 0.717,**無效**。會議系統噪音閘門的密度與辨識品質**無相關**(實測全場均勻) |
| 換 whisper-large-v3 | 0.689,**不如 Breeze**,且輸出簡體 |
| 雲端 OpenAI(gpt-4o-transcribe / whisper-1) | 在最差區段**比地端更糟**:大模型不確定就閉嘴(60 秒吐 9 字),whisper-1 產生「感謝訂閱按讚」的 YouTube 字幕幻覺 |

### ⚠️ 評估陷阱:「輸出變少」會偽裝成「品質變好」

判斷劣化音訊的處理效果時,這個陷阱在一輪實測中**踩了四次**:

- 平均逐字信心:DPDFNet 壓掉六成內容,信心反而**上升**
- 平均逐字信心:large-v3 在最難的一段只吐 5 個字,拿到全場**最高**分 0.726
- RT60:GTCRN 把它從 0.78 壓到 0.49,看似去殘響成功,實際只是削掉低能量尾巴
- CER:DPDFNet 拿下**最低**的 0.450,因為少講就少插入錯誤

**任何指標只要不同時衡量「講了多少」和「講對多少」,就會選出最沉默的那個。** 必須配一個對漏聽敏感的指標(如插入不計分的參考涵蓋率),並實際讀文字。

## 交付檔(存音檔同目錄)

1. `<音檔名>_逐字稿.md` = header + 術語對照表 + `transcript_body.md` 全文。header 含:錄音時長、說話人清單(each: 性別/發言分鐘數/推測身分與**依據**)、產製方式一行、已知誤差(如重疊段)。
2. `<音檔名>_會議摘要與待辦.md`:會議性質一段 → 摘要(依議題分節,節標題帶時間範圍)→ 決議事項 → 待辦表(#/待辦/負責人/時程備註)→ 開放問題 → 結尾一行產製說明。
- **保真原則:逐字稿保留原始聽寫,聽錯也不改寫**;疑似聽錯的專有名詞收進術語對照表(原文→推測正確用語),摘要才用修正後用語。
- 說話人身分靠對話內證推測(誰被點名、誰應答),header 必須標「推測」並給依據。
- 摘要寫完後逐項對照逐字稿核對數字、歸屬、時間範圍;大量條目可發子代理查核(使用者慣例:查證用 opus)。

## 地雷(已寫死在腳本裡,勿改動)

| 事項 | 原因 |
|---|---|
| 聲紋模型必用 `embed_eres2net_common`(20 萬人語料) | sherpa 官方文件範例那顆 `eres2net_base_..._3dspeaker_16k` 是小語料壞模型,4 人會切成 70 人 |
| Breeze 模型下載到預設 HF cache | 自訂深層目錄會爆 Windows 260 字元路徑上限 |
| 標點 API 是 `add_punctuation` | `add_punct` 不存在 |
| 標點後 regex 補中英空格 | 標點模型會吃掉中英之間的空格 |
| transcribe 的抗幻覺參數組(`condition_on_previous_text=False`、三個 threshold=None、repetition_penalty 等) | 2026-07 社群共識,防中文長音檔重複幻覺;勿「順手優化」 |
| CUDA 靠 pip 版 cublas/cudnn/cudart/cufft + `os.add_dll_directory` | 系統未裝 CUDA Toolkit 時的正解 |
| diarize 的 GPU 靠 sherpa-onnx **CUDA wheel**(`==x.y.z+cuda12.cudnn9`,setup.ps1 依 nvidia-smi 自動選) | pip 預設 CPU wheel;`provider="cuda"` 在 CPU wheel 上會**靜默 fallback CPU 不報錯**——diarize 速度異常慢時先用 nvidia-smi 看 VRAM 有無被吃 |
| **同時只跑一個 GPU diarize 程序**;diarize 分塊(10 分/塊)勿改回整檔單體 | 兩程序 VRAM 疊加會原生崩潰(exit 58);單體跑 2h 檔在內部 embedding 階段 MemoryError;sherpa 對每滑窗抽聲紋,embedding 改 CPU 會慢 4 倍——遇 VRAM 問題縮 DIAR_BLOCK,別動 provider |
| 解碼用 faster-whisper 內建 PyAV | 本機沒有 ffmpeg |

## 環境與時效

venv 與 diarization/標點模型在 `%LOCALAPPDATA%\meeting-minutes\`(setup.ps1 管理);ASR 模型在預設 HuggingFace cache。選型是 2026-07 經 4 路上網研究定案(Breeze-ASR-25 在台灣華語/中英夾雜勝過 whisper-large-v3);**若已時隔一年以上,先花 10 分鐘查證有無更好的 zh-TW ASR/diarization 再跑**。
