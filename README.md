# Wash Hand Detect — MediaPipe 七步洗手即時辨識系統

本專案使用 **MediaPipe 雙手骨架追蹤 + OpenCV + 160維幾何拓撲特徵工程 + 15幀時序統計量 (960維) + XGBoost 姿勢分類器 + 觀測狀態與計時分離狀態機 + 統一推論管線 (WashHandPipeline)**，實現標準洗手七步（內、外、夾、弓、大、立、腕）之即時偵測、步驟引導、計時評估與視覺化展示系統。

👉 詳細更新歷程請參閱：[版本更新與修正記錄 (CHANGELOG.md)](./CHANGELOG.md)  
👉 詳細強健度審核與消融實驗報告請參閱：[強健度審查報告 (docs/robustness_review.md)](./docs/robustness_review.md)

---

## 🌟 系統特色

1. **統一推論管線 (`WashHandPipeline`)**：所有即時視訊、離線測試與基準評估共用同一管線引擎，確保等方性座標轉換、特徵抽取與時序平滑契約完全一致。
2. **多維幾何特徵工程 (160 維)**：包含非對稱指節屈曲差值 (`curl_diff`)、Wrist Ratio ($R_{wrist}$)、指節到掌心距離 (Knuckle-to-Palm)、對稱交錯深度 (Interlace Depth) 與指尖聚攏度 (Fingertip Spread)。
3. **15 幀時序統計特徵 (960 維)**：在滑動窗口內計算均值、標準差、極值與起終點差值，捕捉動態洗手搓揉軌跡。
4. **真實觀測與 UI 累計計時分離**：手部暫時遺失時的保留狀態（`held` 幽靈幀）不計入有效洗手秒數；達標門檻嚴格以雙手皆可靠檢出之 `observed_times` 為準。
5. **雙模/混合分類器 (Hybrid Classifier)**：
   - **XGBoost 姿勢分類器**：960 維軟輸出模型，提供 8 類動作機率分布。
   - **非對稱幾何規則分類器**：徹底消除「弓 vs 夾」之特徵混淆，在 `sample_v2` 上「夾」與「弓」達 100% 零混淆。
6. **洗手狀態機 (State Machine)**：
   - **自由模式 (Free Mode, 預設)**：支援任意隨機順序搓洗，達標後持續累計完整動作時長，直至七步全數達標。
   - **教學模式 (Sequence Mode)**：依「內 → 外 → 夾 → 弓 → 大 → 立 → 腕」順序引導。
7. **全畫面 0/1/2 隻真實觀測手詳細覆蓋率統計**：精確分析單手、雙手與未檢出影格分布。

---

## 📂 專案結構

```text
wash_hand_detect/
├── README.md                      # 專案簡介與快速上手指南
├── CHANGELOG.md                   # 版本更新歷程
├── ARCHITECTURE.md                # 系統架構與資料流詳細說明
├── requirements.txt               # 相依套件
├── pytest.ini                     # Pytest 配置 (pythonpath = .)
│
├── models/
│   ├── wash_hand_xgb.joblib       # 960 維時序統計 XGBoost 分類器
│   └── wash_hand_xgb_metadata.json # 訓練 Provenance、參數與交叉驗證指標
│
├── data/
│   ├── sample_v2/                 # 最新更新之獨立七步洗手動作影片 (.mov)
│   ├── sample_v1/                 # 獨立基準七步洗手動作影片 (.mov)
│   ├── clips/                     # YouTube 基準測試片段 (video1_yt, video2_yt2)
│   └── annotated_eval_yt.mp4      # 完整標註基準測試影片
│
├── src/
│   ├── pipeline.py                # 統一洗手偵測推論管線 (WashHandPipeline)
│   ├── camera.py                  # 攝影機 / 測試影像來源封裝
│   ├── hand_detector.py           # 雙手骨架擷取、等方 ROI 映射、無狀態救援與狀態品質
│   ├── features.py                # 160 維幾何特徵計算與歸一化
│   ├── temporal_features.py       # 15 幀時序統計量 (960 維) 與骨架增強
│   ├── rule_classifier.py         # 非對稱規則式分類器
│   ├── ml_classifier.py           # XGBoost / Hybrid 機器學習分類器介面
│   ├── accumulator.py             # 遲滯時序機率積分器
│   ├── state_machine.py           # 狀態機 (觀測秒數與顯示秒數分離)
│   ├── ui.py                      # 繁體中文非遮擋 HUD 渲染
│   ├── realtime.py                # WebCam 即時洗手辨識主程式
│   ├── test_video.py              # 批次影片評估與動作標註影片產出工具
│   ├── evaluate_segments.py       # 基準影片區段辨識率與混淆矩陣報告工具
│   └── train_ml.py                # 跨來源 GroupKFold 訓練與中繼資料導出
│
├── docs/
│   └── robustness_review.md       # 系統強健度、根因分析與消融實驗審查報告
│
└── tests/
    ├── test_pipeline.py           # 基礎特徵與分類器單元測試
    └── test_robustness.py         # 座標 Oracle、計時分離、模型 Fallback 回歸測試
```

---

## 🚀 快速開始

### 1. 安裝環境與套件

```bash
# 建立虛擬環境 (建議 Python 3.10 / 3.11)
python3.11 -m venv venv
source venv/bin/activate

# 安裝相依套件 (含 MediaPipe, OpenCV, XGBoost, Joblib)
pip install -r requirements.txt
```

### 2. 執行最新 `sample_v2` 獨立動作評估

```bash
# 測試 sample_v2 夾步 (interlace)
./venv/bin/python src/evaluate_segments.py --video data/sample_v2/夾.mov

# 測試 sample_v2 弓步 (knuckles)
./venv/bin/python src/evaluate_segments.py --video data/sample_v2/弓.mov

# 測試 sample_v2 大拇指 (thumb) 並產出 0/1/2 觀測手分析
./venv/bin/python src/test_video.py --video data/sample_v2/大.mov
```

### 3. 重新訓練 XGBoost 姿勢分類器 (跨來源無洩漏驗證)

```bash
# 整合 sample_v2、sample_v1 與基準片段訓練，並導出 models/wash_hand_xgb_metadata.json
./venv/bin/python src/train_ml.py --aug 1 --estimators 120
```

### 4. 啟動即時 WebCam 辨識

```bash
# 啟動 WebCam 即時洗手辨識 (自由模式)
./venv/bin/python src/realtime.py --guide-mode free
```

---

## 🧪 執行自動化回歸測試

```bash
./venv/bin/pytest tests/
```

