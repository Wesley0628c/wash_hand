# 掌心相對優先判「內」：修改與測試

日期：2026-09-29。本版依使用者要求，提高掌心相對、雙手合攏的「內」證據權重，減少過度拒判。沒有恢復必須同時偵測雙手的全域限制。

## 修改內容

1. **放寬內的幾何條件**：掌面法向量點積 < -0.30、掌心距離 <1.35 掌長，且沒有明確交錯證據時，優先給內 3.4 分。移除手軸 <28°、手腕與掌距差 <0.20、掌距 <0.95 的額外限制。掌面類分支允許較小四指平均角 >120°、較大者 >150°，不要求兩手都完全伸直。弓／大／立／腕的專屬姿態仍可先匹配。
2. **夾不再只看張開**：要求跨手指尖投影至少 4 次左右交替、投影重疊 ≥0.25、四指平均散開 ≥0.30、手軸角 >28°、交錯深度 <1.15。單手張開不直接分類為夾。這仍是二維幾何近似，不能宣稱已完全確認真實三維穿插。
3. **單手平掌改為內候選**：原先四指角 ≥140° 且沒有其他匹配時直接輸出 other，現改給 inside 1.6 分。約 0.414 的 softmax 分數足以顯示候選，但低於現有 0.45 計時門檻，不會只憑單手伸直自動完成內。明確雙手掌面證據分數約 0.81；這是規則分數，不是經校準的正確率。
4. **不直接降低全部門檻**：分類平滑仍為 0.5 秒、顯示門檻仍為 0.30、計時門檻仍為 0.45。查到大量 other 的主要來源是直接拒絕平掌的規則，因此先改規則。
5. **保留短遮擋顯示連續性**：沿用工作目錄已有的 held 座標補償，但增加 `both_hands_observed`、`contains_held` 區分真雙手與歷史補點；只有真雙手使用相對位移計時，單手／held 情況用形變驗證，避免真手平移加舊手座標冒充搓洗。
6. **避免放寬後誤計時**：恢復當前 raw 與顯示類一致才能計時；超過容忍間隔的零碎片段不靠歷史總量湊完成。這兩項只影響進度，不提高 other 比例。
7. **文字**：other 改為「其他（未偵測／動作待確認）」，避免把低信心直接說成非標準動作。

單手候選不需要先出現雙手；其他單手姿態仍可按其分數與動態獨立計時。只有單手平掌時，系統無法確知另一面接觸的是掌心、手背或拇指，所以沒有把該候選提升成高信心完成證據。

## 基準與整體結果

比較本次開始時的工作目錄版本，不是上一份單手報告或 Git HEAD。修改前來源保存在 `scratch/palm_priority_before/`。同一份 `scratch/sample_detections.json`，重播 14 支影片、2689 幀；全畫面、rule/free、1 秒達標。檔名是片段標籤，包含轉場，兩組樣本是開發資料而非獨立測試集。

| 指標 | 修改前 | 修改後 |
|---|---:|---:|
| 正確分類／全部幀 | 18.7% | 27.7% |
| 錯誤洗手類／全部幀 | 15.8% | 33.5% |
| other／全部幀 | 65.6% | 38.9% |
| 顯示內／全部幀（包含誤判） | 8.1% | 36.1% |
| 錯誤完成事件 | 2 | 1 |

**降低拒判並不等於全面提高正確率。** 內的召回明顯改善，但部分外、夾、大會被單手平掌候選吸收到內，錯誤洗手類比例也增加。故保留較弱候選與有效計時的區分。sample_v2/外 仍發生內的錯誤完成，顯示 MediaPipe 掌面朝向與目前近似接觸條件仍會混淆掌心／手背。

## 逐片正確分類比例

| 影片 | 修改前 | 修改後 |
|---|---:|---:|
| sample_v1/內.mov | 6.6% | 74.2% |
| sample_v1/外.mov | 7.0% | 0.0% |
| sample_v1/夾.mov | 8.7% | 8.2% |
| sample_v1/弓.mov | 2.3% | 2.3% |
| sample_v1/大.mov | 0.0% | 0.0% |
| sample_v1/立.mov | 55.7% | 58.6% |
| sample_v1/腕.mov | 0.0% | 0.0% |
| sample_v2/內.mov | 14.9% | 97.3% |
| sample_v2/外.mov | 37.5% | 16.2% |
| sample_v2/夾.mov | 0.0% | 0.0% |
| sample_v2/弓.mov | 22.9% | 23.4% |
| sample_v2/大.mov | 20.1% | 20.1% |
| sample_v2/立.mov | 42.3% | 42.3% |
| sample_v2/腕.mov | 49.7% | 56.2% |

## 完整影片結果

兩支都由原始影片重新偵測，未將已有骨架的成品再送入分類。

| 影片 | 讀取幀數 | other | 內 | 夾 | 完成事件 |
|---|---:|---:|---:|---:|---|
| video1 | 1147 | 661（57.6%） | 295（25.7%） | 5（0.4%） | 31.13s fingertips |
| video2 | 971 | 377（38.8%） | 140（14.4%） | 8（0.8%） | 13.68s inside、38.29s wrist |

使用者提供的統計為 970 幀；video2 此次讀取 971 幀，不能宣稱輸入與參數完全相同。可確認此版 video2 顯示 other 為 38.8%、內為 14.4%。

抽查 [結果截圖](assets/palm_priority/comparison.jpg)：video1 的 3.2 秒「內」示範已顯示內；但 video2 的 13.7 秒在「外」示範仍判為內，且剛完成內。這是尚未解決的掌背混淆，完成數不能當作正確率。

## 回歸測試

完整測試 **43 passed**。新增案例包含掌心相對不受舊手軸／腕距限制、沒有交替排列的張開手指不能當夾、交錯證據可勝過內、同向掌面不自動算內、遠距雙手與無手仍拒判、單手拇指張開不會觸發夾。原有靜止、轉場與長中斷計時檢查保留。

既有 held 測試改為驗證 `both_hands_observed=False` 與 `contains_held=True`，因為工作目錄已允許舊座標參與短暫顯示補償；測試仍要求只有平移的真手加 held 手不得取得進度。

## 結果與重跑

- [修改前統計](evaluation/palm_before.json)／[逐幀資料](evaluation/palm_before.csv)
- [修改後統計](evaluation/palm_after.json)／[逐幀資料](evaluation/palm_after.csv)
- [第一支完整影片](../data/processed/wash_7steps_palm_priority.mp4)／[逐幀紀錄](evaluation/palm_video1.jsonl)
- [第二支完整影片](../data/processed/wash_7steps_yt2_palm_priority.mp4)／[逐幀紀錄](evaluation/palm_video2.jsonl)

```bash
venv/bin/python -m pytest -q
venv/bin/python src/evaluate_samples.py --output docs/evaluation/palm_after
venv/bin/python src/test_video.py --video data/raw/wash_7steps_yt.mov --ori 1.5 --max-frames 0 --output data/processed/wash_7steps_palm_priority.mp4 --trace docs/evaluation/palm_video1.jsonl
venv/bin/python src/test_video.py --video data/raw/wash_7steps_yt2.mov --max-frames 0 --output data/processed/wash_7steps_yt2_palm_priority.mp4 --trace docs/evaluation/palm_video2.jsonl
```

執行包含 MediaPipe 的測試及影片處理時，macOS 需可建立 OpenGL context 的本機環境。
