# 「外」的接觸位置、滑動方向與誤判修正

日期：2026-09-29。比較基準是本次修改前的掌心優先版本（`palm_after`），不是 Git HEAD。

## 結果先說

已修改分類、接觸特徵、角色追蹤與計時條件，並回放兩組樣本、重跑兩支原始完整影片。

- `sample_v2/外` 的顯示召回率 **16.3% → 30.6%**；在第二組全部七類樣本計算，「外」precision **57.8% → 100%**、F1 **25.4% → 46.9%**。100% 只代表這組開發資料中的 49 次「外」預測均落在外片段，不能外推成實際準確率。
- 兩組「外」被顯示成「內」共 **186 → 27 幀**。
- `sample_v1/外` 仍 **0/257 幀辨識正確**。完整影片也沒有完成「外」。**尚未完整修復。**
- video2 在 13.68 秒錯誤完成「內」的事件已消除；13.7 秒的顯示仍是「內」，但暫停計時。這次修正了該處的誤完成，沒有把它變成正確的外辨識。
- 代價：樣本整體 `other` **38.9% → 50.5%**，整體正確分類 **27.7% → 27.3%**；錯誤洗手類 **33.5% → 22.3%**。本版較保守，不能宣稱全面準確率提高。

## 1. 原本哪裡造成誤判

### 1.1 內先匹配，外無法競爭

原本平掌分支先檢查 `palm_normal_dot < -0.30` 與掌距 `<1.35`，成立即给 inside 3.4 分；外另有 `not is_inside` 限制。法向量一旦落入內的範圍，即使有手背滑動，外也沒有獨立證據可以勝出。

在快取樣本中，`sample_v1/外` 真雙手影格的 70%、`sample_v2/外` 的 86.4%，法向量都落入這個內的範圍。朝向估計、左右標籤及遮擋都有可能影響結果；這些數據不能單獨證明是哪個偵測器環節錯誤。

### 1.2 只有一隻新觀測手，也被當作強雙掌證據

pipeline 允許新觀測手與 held 舊座標一起抽特徵，以維持顯示。原分類器未區分此情況，仍可能給 inside 3.4。video2 的 13.68 秒就是 `left=held, right=observed`，舊版仍給內有效時間，最後誤完成。

因此，降低純單手 1.6 分還不夠，**held 雙手分支也必須降權**。

### 1.3 外缺乏接觸區域與運動檢查

旧外規則依靠掌面、掌距、腕距差及手軸角；沒有確認接觸在掌背中央，也沒有確認相對滑動沿手背方向。單手拇指收攏的平掌亦可成為外候選。

### 1.4 不是完成後又自動偏向內

目前 `get_state_prior()` 為中性，pipeline 沒有使用完成進度提高 inside 分數。13.68 秒是錯誤證據累積後的完成事件；加「完成後 0.5 秒冷卻」無法阻止這次首次誤完成。本次改證據權重，不加入完成狀態對分類的偏差。

## 2. 定義與可觀測限制

[WHO 洗手圖示](https://www.who.int/docs/default-source/documents/health-topics/hand-hygiene-why-how-and-when-brochure.pdf) 區分掌心相對、掌心覆在對側手背，以及掌心相對時手指交錯。**外也可以有交錯手指**，所以不能把所有交錯直接排除為非外。本專案沿用「內外夾弓大立腕」標籤；WHO 這份圖示不應被描述成相同的七類資料標註。

[MediaPipe 官方文件](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md?plain=1) 說明每隻手的 landmark z 以自己的手腕為原點。不能把兩隻手的 z 直接當成共用深度來確認表面接觸。

掌心和手背也沒有各自獨立的 landmark 中心。因此本次新增的是 **影像平面接觸區域＋滑動＋角色的估計**，不是已經量出皮膚接觸面。法向量只提供輔助分數，沒有單獨決定內外。側視、交疊及錯誤骨架仍可能使估計失敗。

## 3. 現在的判斷流程

### 3.1 內、外候選分數

這些是 softmax 前的規則 logit，不是正確率，也不是接觸分數。

| 證據 | 修改前 | 修改後 |
|---|---:|---:|
| 純單手平掌 | inside 1.6 | inside 1.2，約 0.322 的正規化分數 |
| 雙手掌面相對且接近 | inside 3.4 | inside 3.2 |
| 一手 observed、一手 held，匹配內 | inside 3.4 | 上限 1.6，只有顯示候選強度 |
| 只有舊外朝向／腕距條件 | outside 3.4 | outside 1.8，弱候選 |
| 新接觸與滑動證據 ≥0.60 | 無 | outside 3.4，獨立參與競爭 |

當新外證據成立且外 logit ≥ 內 logit −0.3，內扣 0.8 分。**沒有接觸與滑動證據，不套用這個扣分。** 舊單手外形態候選仍保留 2.1 分；它不能單憑該分數取得外的有效時間。

腕、弓、大、立的既有專屬幾何分支保留；新外證據另檢查平掌、接觸區域與負面條件。單手辨識仍可執行，沒有恢复必須同時偵測雙手的全域開關。

### 3.2 分開計算兩個角色

新增 `src/back_contact.py`，兩方向分別為 `left_palm_right_back`、`right_palm_left_back`。名稱代表估計角色，不能理解成表面已確認。

對目標手取四個 MCP 平均，建立 `u = MCP中心 − 手腕`。接觸點代理值是作用手的「手腕與 MCP 中心中點」。在目標手座標系表示接觸位置：

- 沿手軸 `<0.15` 掌長：wrist。
- 沿手軸 `>1.15` 掌長：fingers。
- 中間範圍但橫向超過 `0.75 × 掌寬`：off_hand。
- 其餘：central，只能確認投影在中央區域，不能直接分出掌／背。

### 3.3 相對滑動與分數

用同一角色的前後接觸點，在目標手自己的旋轉／縮放座標系計算位移，再除以真實 dt。共同平移、旋轉、縮放不會被當成相對摩擦；跨手 z 不參與這項計算。

有效滑動要求：前後都有當前真雙手觀測、速度在 **0.04～12 掌長／秒**、與目標手軸的方向相似度絕對值 **>0.5**。重新取得追蹤的第一幀不建立運動證據。

外接觸分數（裁切至 0～1）：

```text
0.40 × 中央接觸分數
+ 0.25 × 有效滑動
+ 0.20 × 軸向相似度 × 有效滑動
+ 0.15 × 作用手移動大於目標手的程度
+ 0.10 × 法向量提供的掌背方向估計
− 0.12 × 掌面相對且運動對稱的程度
− 0.25 × 掌面相對且有交错重疊
− 0.35 × 拇指包覆證據
− 0.25 × 指尖集中接觸掌心證據
```

還須通過平掌與 central 接觸條件。接觸靠腕、靠指、離開手部、沒有相對滑動、追蹤跳動都不能建立新的外證據。對稱性、方向及負面分數只是啟發式；掌心相對的縱向摩擦仍可能與外相似。

### 3.4 角色穩定、短遮擋與計時

- 每個角色分開評分，取最高者；若既有角色與最高分差 `<0.10`，保留既有角色，減少近似平手時跳換。
- 外的有效計時要求：接觸分數 ≥0.60、角色維持 ≥0.40 秒、最近 0.80 秒內有 ≥0.50 秒新觀測滑動證據。
- 超過 0.15 秒沒有有效接觸，或真正更換角色，重置角色歷史；這不是嚴格零間斷的 0.5 秒，允許短暫轉向／遮擋，但不把缺失時間補算為滑動。
- 短遮擋最多延續 0.15 秒；可由單手延續已建立的外證據，不能由 held 座標憑空建立新的接觸或增加 motion_seconds。只有目前可見平掌且形變合格時，才有機會延續計時。
- 完全只出現單手、沒有可用的歷史接觸時，仍可顯示候選，但目前骨架規則無法確認它正在接觸哪個表面，因此不讓外完成。
- 仍需通過 pipeline 原有 raw/display 一致、信心 ≥0.45、類間差 ≥0.12、動態及時間支持檢查。最後累積有效時間 1.0 秒才完成；較長中斷仍沿用狀態機的清除規則。

`outside_min_score`、`outside_role_seconds`、`outside_entry_seconds`、`outside_occlusion_seconds` 放在 `PipelineConfig`。0.60 是接觸特徵分數門檻，不能當成 softmax 機率 60%。

## 4. 樣本測試與混淆矩陣

同一份 `scratch/sample_detections.json` 回放 14 支影片、2689 幀。兩組都用相同參數；未依 video1/video2 檔名切换門檻。既有 160 維 ML 特徵排列不變，本次主要驗證預設 rule 模式。

**標註是檔名的片段類別，包含轉場，並非人工逐幀精確標註；資料也是調整規則所用的開發樣本，沒有獨立測試集。** 新增 raw/display/credit 三種逐幀矩陣和每類 precision、recall、F1。無預測時 precision 寫 null，而不是假稱 100%。

| 外的指標（display） | 修改前 | 修改後 |
|---|---:|---:|
| sample_v1 recall | 0.0% | 0.0% |
| sample_v1 precision | 無外預測 | 無外預測 |
| sample_v2 precision | 57.8% | 100.0% |
| sample_v2 recall | 16.3% | 30.6% |
| sample_v2 F1 | 25.4% | 46.9% |
| 兩組合併 precision | 57.8% | 100.0% |
| 兩組合併 recall | 6.2% | 11.8% |
| 兩組合併 F1 | 11.3% | 21.0% |

「內」的合併 precision 32.3% →42.5%，recall 83.3% →72.7%。其中 sample_v1/內 recall 74.2% →68.1%，sample_v2/內 97.3% →79.7%。抑制假雙掌證據也損失了部分被遮擋的真內。

| 外片段被顯示為內 | 修改前 | 修改後 |
|---|---:|---:|
| sample_v1/外 | 99 | 18 |
| sample_v2/外 | 87 | 9 |

兩組外的 **credit recall 仍為 0%**，沒有外完成事件；顯示改善不等於計時改善。sample_v2/外 的最大有效滑動支持只有約 0.252 秒，未達進入條件 0.50 秒。樣本錯誤完成事件由 1 次（外片段完成內）降為 0 次；正確完成仍只有 sample_v1/立。

`sample_v1/外` 257 幀中：10 幀 observed/observed、84 幀只有一手 observed 且另一手 missing、51 幀一手 observed 一手 held，其餘 112 幀沒有新觀測。只有這 10 幀的零碎雙手觀測，沒有產生符合前後連續性與方向條件的外滑動證據。降低 inside 只能讓它變成待確認，不能補出被遮住的掌背表面。

## 5. 原始完整影片

由 `data/raw` 重跑，沒有把帶 HUD／骨架的成品再次拿去辨識。完整影片未有可靠逐幀真值，因此不報它們的 precision/recall。

| 影片／版本 | 幀數 | 顯示 other | 顯示內 | 顯示外 | 完成事件 |
|---|---:|---:|---:|---:|---|
| video1 修改前 | 1147 | 661（57.6%） | 295 | 40 | 31.13s 立 |
| video1 修改後 | 1147 | 799（69.7%） | 164 | 27 | 31.10s 立 |
| video2 修改前 | 971 | 377（38.8%） | 140 | 114 | 13.68s 內（錯誤）、38.29s 腕 |
| video2 修改後 | 971 | 487（50.2%） | 78 | 50 | 38.20s 腕 |

兩支完整影片修改後都是 1/7 完成。video2 的內誤完成消失，但約 13.7 秒仍顯示內、`credit=other / ambiguous_scores`；見 [畫面比較](assets/back_contact/comparison.jpg)。完成事件數不代表洗手品質或分類準確率。

## 6. 測試、檔案與重跑

完整回歸 **62 passed**。新增檢查涵蓋：共同平移／旋轉／縮放、不同手部 z 原點、15/30/60 FPS、縱向與橫向滑動、接觸區域、角色穩定及切換、held／重新追蹤不能建立新證據、單手短延續、靜止與腕接觸拒絕計時、長中斷、內外分數競爭、held 內的降權。macOS 的兩項 MediaPipe 測試需本機 OpenGL；沙盒內因無法建立 context 失敗，改在可使用 OpenGL 的環境重跑完整測試後通過。

- [核心接觸特徵與角色追蹤](../src/back_contact.py)、[分類規則](../src/rule_classifier.py)、[管線與計時](../src/pipeline.py)
- [樣本修改前矩陣](evaluation/back_contact_before.metrics.json)、[修改後矩陣](evaluation/back_contact_after.metrics.json)
- [逐片結果](evaluation/back_contact_after.json)、[逐幀 CSV](evaluation/back_contact_after.csv)、[完整影片統計](evaluation/back_contact_full_summary.json)
- [video1 成品](../data/processed/wash_7steps_back_contact.mp4)、[video2 成品](../data/processed/wash_7steps_yt2_back_contact.mp4)
- [video1 trace](evaluation/back_contact_video1.jsonl)、[video2 trace](evaluation/back_contact_video2.jsonl)

```bash
MPLCONFIGDIR=/tmp/wash_mpl venv/bin/python -m pytest tests -q
MPLCONFIGDIR=/tmp/wash_mpl venv/bin/python src/evaluate_samples.py --output docs/evaluation/back_contact_after
venv/bin/python src/evaluation_metrics.py docs/evaluation/palm_after.csv --output docs/evaluation/back_contact_before.metrics.json
MPLCONFIGDIR=/tmp/wash_mpl venv/bin/python src/test_video.py --video data/raw/wash_7steps_yt.mov --ori 1.5 --max-frames 0 --output data/processed/wash_7steps_back_contact.mp4 --trace docs/evaluation/back_contact_video1.jsonl
MPLCONFIGDIR=/tmp/wash_mpl venv/bin/python src/test_video.py --video data/raw/wash_7steps_yt2.mov --max-frames 0 --output data/processed/wash_7steps_yt2_back_contact.mp4 --trace docs/evaluation/back_contact_video2.jsonl
```

## 7. 下一步應如何改

1. **先補能區分掌背的輸入**：對手部影像裁切建立掌心／手背／不確定分類器，並標記遮擋。用影像外觀與骨架位置共同判斷接觸，避免只看骨架法向量。單手畫面若要獨立辨識外，這比繼續放寬骨架分數更值得驗證。
2. **改善手部身份與遮擋追蹤**：檢查左右 ID 跳換、重複偵測同一隻手、短暫消失；加入可評估品質的影像追蹤，再考慮放寬角色支持。不要讓 held 自己製造新的摩擦證據。
3. **人工標註完整段落與負例**：至少區分真內、真外、只是靠近、靜止、轉場及不可判。補足其他人、角度、光線與泡沫的獨立資料，再做逐幀混淆矩陣。
4. **用分組驗證選共同參數**：依人／影片切開訓練與驗證，搜尋接觸、運動、時間門檻，同時約束外 precision、內 recall、other 比例與錯誤完成事件。兩支影片的結果應分開報告，但不根據檔名套不同門檻。現在兩組都是開發資料，未做會被誤解為泛化能力的參數搜尋。

本版落實接觸、方向、角色與計時保護，但純骨架下的掌背不可辨識與高遮擋仍是限制。後續目標應是補足辨識證據，使 other 下降且維持低誤判，而不是靠更高固定分數湊成 7/7。
