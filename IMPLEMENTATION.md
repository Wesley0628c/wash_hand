# Wash Hand Detect — MediaPipe 七步洗手辨識實作指南

本專案目標是使用 **MediaPipe + OpenCV + 時序模型**，即時辨識洗手七個標準動作：

1. 內：掌心對掌心搓洗
2. 外：掌心搓洗手背
3. 夾：十指交錯搓洗
4. 弓：手指彎曲，以指背搓洗另一手掌
5. 大：旋轉搓洗大拇指
6. 立：指尖搓洗另一手掌心
7. 腕：旋轉搓洗手腕

另外建議增加第 8 類：

- other：非標準洗手動作、尚未開始洗手、手勢不完整

---

## 1. 系統架構

```text
Webcam
  ↓
OpenCV 擷取影像
  ↓
MediaPipe Hand Landmarker
  ↓
左右手各 21 個 landmarks
  ↓
Landmark normalization
  ↓
Feature Extraction
  ├─ 左右手座標
  ├─ 手指角度
  ├─ 雙手距離
  ├─ Palm Center
  ├─ Palm Normal
  └─ Movement / Velocity
  ↓
最近 30 幀 Sliding Window
  ↓
分類器
  ├─ MVP：Rule-based
  └─ 正式版：LSTM / GRU
  ↓
Temporal Smoothing
  ↓
State Machine
  ↓
即時顯示
```

---

## 2. 建議技術

- Python 3.10 / 3.11
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib

安裝：

```bash
python -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python numpy pandas scikit-learn tensorflow matplotlib
```

Windows：

```bash
venv\Scripts\activate
```

---

## 3. 建議專案結構

```text
wash_hand_detect/
├── README.md
├── IMPLEMENTATION.md
├── requirements.txt
│
├── models/
│   ├── hand_landmarker.task
│   └── wash_hand_lstm.keras
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── labels.csv
│
├── src/
│   ├── camera.py
│   ├── hand_detector.py
│   ├── features.py
│   ├── rule_classifier.py
│   ├── collect_data.py
│   ├── train.py
│   ├── realtime.py
│   └── state_machine.py
│
└── notebooks/
    └── analyze_landmarks.ipynb
```

---

# Phase 1：先把 Webcam 跑起來

先確認 OpenCV 可以正常讀取攝影機。

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Wash Hand Detect", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

完成標準：

- Webcam 正常開啟
- FPS 穩定
- 按 q 可以正常離開

---

# Phase 2：加入 MediaPipe Hand Landmarker

洗手需要兩隻手，因此必須設定：

```text
num_hands = 2
```

每一隻手有 21 個 landmark：

```text
0  Wrist

Thumb
1  CMC
2  MCP
3  IP
4  TIP

Index
5  MCP
6  PIP
7  DIP
8  TIP

Middle
9  MCP
10 PIP
11 DIP
12 TIP

Ring
13 MCP
14 PIP
15 DIP
16 TIP

Pinky
17 MCP
18 PIP
19 DIP
20 TIP
```

每個 landmark 包含：

```text
x, y, z
```

所以：

```text
單手：21 × 3 = 63 features
雙手：42 × 3 = 126 features
```

---

# Phase 3：Landmark Normalization

不要直接將原始畫面座標丟進模型。

否則同一個姿勢只因為：

- 手在畫面左邊
- 手在畫面右邊
- 手靠近鏡頭
- 手遠離鏡頭

就可能被模型當成不同姿勢。

## 3.1 Wrist 當原點

```python
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    normalized = []

    for point in landmarks:
        normalized.append([
            point[0] - wrist[0],
            point[1] - wrist[1],
            point[2] - wrist[2],
        ])

    return normalized
```

## 3.2 使用手掌大小做 scale normalization

可使用：

```text
Wrist → Middle MCP
```

即：

```text
landmark 0 → landmark 9
```

```python
import numpy as np

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))
```

接著：

```python
scale = distance(landmarks[0], landmarks[9])
```

所有相對座標再除以 scale。

---

# Phase 4：左右手順序固定

模型輸入必須永遠固定：

```text
Left Hand
+
Right Hand
```

不能某一幀：

```text
Left + Right
```

下一幀卻變：

```text
Right + Left
```

若某隻手沒有偵測到：

```python
ZERO_HAND = [0.0] * 63
```

以 0 補滿。

---

# Phase 5：Feature Extraction

洗手七步最大的重點是：

> 不只是看單手形狀，而是看兩隻手之間的互動。

因此除了 126 個 landmark features，還應加入以下資訊。

---

## 5.1 雙手 Wrist 距離

```python
wrist_distance = distance(
    left_hand[0],
    right_hand[0]
)
```

可以判斷雙手是否靠近。

---

## 5.2 指尖距離

重要指尖：

```python
TIP_IDS = [4, 8, 12, 16, 20]
```

計算左右對應指尖距離。

---

## 5.3 Palm Center

建議用：

```python
PALM_IDS = [0, 5, 9, 13, 17]
```

```python
def palm_center(hand):
    points = np.array([hand[i] for i in PALM_IDS])
    return points.mean(axis=0)
```

對「立、大、腕」尤其重要。

---

## 5.4 Palm Normal

使用：

- Wrist
- Index MCP
- Pinky MCP

估計掌面方向。

```python
def palm_normal(hand):
    wrist = np.array(hand[0])
    index = np.array(hand[5])
    pinky = np.array(hand[17])

    v1 = index - wrist
    v2 = pinky - wrist

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm == 0:
        return normal

    return normal / norm
```

左右掌面的關係：

```python
dot = np.dot(
    palm_normal(left_hand),
    palm_normal(right_hand)
)
```

可幫助區分：

- 內
- 外

---

## 5.5 Finger Angle

計算三點夾角：

```python
def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
        + 1e-8
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))
```

對「弓」尤其重要，因為該動作手指會明顯彎曲。

---

## 5.6 Movement / Velocity

洗手七步是動作，不是七張靜態圖片。

因此必須計算：

```text
Current frame - Previous frame
```

例如：

```python
velocity = current_landmarks - previous_landmarks
```

如此才能分辨：

```text
只是把手放著
```

和：

```text
真的正在摩擦 / 旋轉
```

---

# Phase 6：七個動作如何判斷

## 6.1 內

特徵：

- 雙手距離近
- 掌心相對
- 指尖方向接近
- 有來回摩擦 movement

概念：

```python
if (
    hands_close
    and palms_facing_each_other
    and rubbing_motion
):
    gesture = "inside"
```

---

## 6.2 外

特徵：

- 一手掌心貼另一手背
- 兩手距離近
- 掌面方向與「內」不同
- 有往返摩擦 movement

需要辨識：

```text
左掌 → 右手背
右掌 → 左手背
```

---

## 6.3 夾

特徵：

- 十指交錯
- 左手指尖進入右手手指區域
- 右手指尖進入左手手指區域
- 兩手距離非常近
- 有左右摩擦 movement

---

## 6.4 弓

特徵：

- 手指 PIP / DIP 明顯彎曲
- 指背靠近另一手 Palm Center
- 有摩擦 movement

這類通常最容易受到雙手遮擋影響。

---

## 6.5 大

特徵：

- 一手包住另一手 Thumb
- finger landmarks 集中在另一手拇指附近
- 具有旋轉 movement

Thumb landmarks：

```text
1 CMC
2 MCP
3 IP
4 TIP
```

---

## 6.6 立

特徵：

- Index / Middle / Ring / Pinky 指尖集中
- 指尖中心靠近另一手 Palm Center
- 有旋轉或來回 movement

```python
TIP_WITHOUT_THUMB = [8, 12, 16, 20]

def finger_tip_center(hand):
    points = np.array([
        hand[i]
        for i in TIP_WITHOUT_THUMB
    ])

    return points.mean(axis=0)
```

---

## 6.7 腕

特徵：

- 一手包住另一手 Wrist
- 一手 Palm Center 靠近另一手 landmark 0
- 有旋轉 movement

例如：

```python
distance(
    right_palm_center,
    left_wrist
)
```

很小且有旋轉動作，可視為 wrist candidate。

---

# Phase 7：第一版先做 Rule-based

不要一開始就做 LSTM。

先建立：

```text
src/rule_classifier.py
```

架構：

```python
class WashHandRuleClassifier:

    def predict(self, features):

        if self.is_inside(features):
            return "inside"

        if self.is_outside(features):
            return "outside"

        if self.is_interlace(features):
            return "interlace"

        if self.is_knuckles(features):
            return "knuckles"

        if self.is_thumb(features):
            return "thumb"

        if self.is_fingertips(features):
            return "fingertips"

        if self.is_wrist(features):
            return "wrist"

        return "other"
```

先用規則版找出：

- 哪些類最好辨識
- 哪些 landmark 最容易飄
- 哪些類容易互相混淆
- 攝影機角度是否適合

---

# Phase 8：蒐集訓練資料

分類建議使用 8 類：

```text
other
inside
outside
interlace
knuckles
thumb
fingertips
wrist
```

## 建議至少

```text
10～20 人
```

每人每個洗手動作：

```text
10～20 次
```

例如：

```text
20 人 × 7 動作 × 10 次
= 1400 clips
```

另外需要大量 other samples：

- 揮手
- 握拳
- 張手
- 隨便搓手
- 雙手靠近
- 拿物品
- 調整衣服
- 手放桌上

否則模型看到任何未知動作時，仍會硬猜七步中的其中一個。

---

# Phase 9：Sliding Window

假設 Webcam 是 30 FPS。

建議每個 sample 使用：

```text
30 frames
```

約等於 1 秒。

模型輸入：

```text
30 × feature_dimension
```

若只用 landmarks：

```text
30 × 126
```

加入 angles、distance、normal、velocity 後，可能會變成：

```text
30 × 150～180
```

---

# Phase 10：資料儲存

建議使用 NumPy：

```text
.npy
```

例如：

```text
data/processed/
├── person_01/
│   ├── inside_001.npy
│   ├── outside_001.npy
│   ├── interlace_001.npy
│   ├── knuckles_001.npy
│   ├── thumb_001.npy
│   ├── fingertips_001.npy
│   ├── wrist_001.npy
│   └── other_001.npy
└── person_02/
```

---

# Phase 11：Train / Validation / Test 必須依照人分割

不要把所有影片打散再切 80/20。

錯誤方式：

```text
同一個人同時出現在 Train 和 Test
```

這容易造成 Data Leakage。

建議：

```text
Person 01～14 → Train
Person 15～17 → Validation
Person 18～20 → Test
```

如此才能真正測試：

> 新使用者第一次出現在系統時，模型還能不能正確辨識。

---

# Phase 12：LSTM 模型

範例：

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(30, feature_dim)
    ),

    tf.keras.layers.LSTM(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        8,
        activation="softmax"
    )
])
```

Compile：

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

# Phase 13：Label Map

```python
LABELS = {
    0: "other",
    1: "inside",
    2: "outside",
    3: "interlace",
    4: "knuckles",
    5: "thumb",
    6: "fingertips",
    7: "wrist",
}
```

中文顯示：

```python
LABEL_ZH = {
    "other": "其他",
    "inside": "內",
    "outside": "外",
    "interlace": "夾",
    "knuckles": "弓",
    "thumb": "大",
    "fingertips": "立",
    "wrist": "腕",
}
```

---

# Phase 14：即時 Prediction

使用：

```python
from collections import deque

sequence = deque(maxlen=30)
```

每個 frame：

```python
sequence.append(features)
```

當：

```python
len(sequence) == 30
```

再呼叫模型。

---

# Phase 15：Confidence Threshold

不要只看最大機率。

例如：

```python
confidence = prediction.max()
label = prediction.argmax()

if confidence < 0.80:
    label = "other"
```

可以實驗：

```text
0.70
0.75
0.80
0.85
0.90
```

---

# Phase 16：Temporal Smoothing

避免輸出：

```text
內
內
外
內
內
```

一直閃爍。

可保留最近 10 次 prediction：

```python
prediction_history = deque(maxlen=10)
```

例如規定：

```text
最近 10 次有 8 次相同
```

才正式切換動作。

---

# Phase 17：State Machine

即時洗手監測需要狀態機：

```text
WAITING
↓
INSIDE
↓
OUTSIDE
↓
INTERLACE
↓
KNUCKLES
↓
THUMB
↓
FINGERTIPS
↓
WRIST
↓
COMPLETE
```

可提供兩種模式。

## 教學模式

強制順序：

```text
內 → 外 → 夾 → 弓 → 大 → 立 → 腕
```

## 自由模式

不限制順序，但七項最後都需要完成。

---

# Phase 18：持續時間判定

模型辨識出一次不能直接算完成。

例如：

```text
正確動作持續 2～3 秒
```

才算完成。

```python
if label == current_step:
    step_timer += delta_time

    if step_timer >= 3.0:
        complete_step()
```

可以允許短暫 0.2～0.5 秒的 prediction error，不必立刻歸零。

---

# Phase 19：UI

建議顯示：

```text
┌──────────────────────────────┐
│                              │
│      MediaPipe Skeleton      │
│                              │
│ Current：夾                  │
│ Confidence：93%              │
│                              │
│ ✅ 內                        │
│ ✅ 外                        │
│ ▶ 夾  1.8 / 3 sec            │
│ ○ 弓                         │
│ ○ 大                         │
│ ○ 立                         │
│ ○ 腕                         │
│                              │
│ Total Time：8.6 sec          │
└──────────────────────────────┘
```

---

# Phase 20：即時錯誤提示

除了分類結果，規則系統還能產生 feedback：

```text
雙手距離太遠
```

```text
請讓十指交錯
```

```text
請把手指彎曲
```

```text
請讓指尖接觸掌心
```

```text
姿勢正確，請繼續搓洗
```

推薦做法：

```text
LSTM / GRU
→ 負責動作分類

Rule-based features
→ 負責錯誤原因與提示
```

---

# Phase 21：模型評估

不要只看 Accuracy。

至少輸出：

- Precision
- Recall
- F1-score
- Confusion Matrix

```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_true, y_pred))

cm = confusion_matrix(
    y_true,
    y_pred
)
```

特別觀察：

```text
內 vs 外
夾 vs 弓
大 vs 腕
```

---

# Phase 22：建議實作順序

## Milestone 1：MediaPipe Pipeline

- [ ] Webcam 正常
- [ ] 同時偵測兩手
- [ ] 顯示 42 個 landmarks
- [ ] 左右手可穩定區分
- [ ] Landmark normalization
- [ ] Palm Center
- [ ] Palm Normal
- [ ] Finger Angle
- [ ] Hand Distance
- [ ] Movement Velocity

## Milestone 2：Rule-based MVP

- [ ] 內
- [ ] 外
- [ ] 夾
- [ ] 弓
- [ ] 大
- [ ] 立
- [ ] 腕
- [ ] other

## Milestone 3：資料收集

- [ ] collect_data.py
- [ ] 10～20 位測試者
- [ ] 每動作 10～20 clips
- [ ] 大量 other samples
- [ ] 30-frame sequence
- [ ] 依 person 分 Train / Validation / Test

## Milestone 4：時序模型

- [ ] LSTM / GRU
- [ ] 8 classes
- [ ] Confidence Threshold
- [ ] Temporal Smoothing
- [ ] Confusion Matrix
- [ ] F1-score

## Milestone 5：完整應用

- [ ] State Machine
- [ ] 步驟完成計時
- [ ] 七步進度 UI
- [ ] 即時錯誤提示
- [ ] 完成後顯示洗手結果

---

# 最終推薦架構

```text
Camera
  ↓
MediaPipe
  ↓
42 Hand Landmarks
  ↓
Normalization
  ↓
Feature Extraction
  ↓
30-frame Sequence
  ↓
LSTM / GRU
  ↓
8-class Softmax
  ↓
Confidence Threshold
  ↓
Temporal Smoothing
  ↓
State Machine
  ↓
Wash Hand Feedback
```

最重要的觀念：

> MediaPipe 負責找出手部骨架；真正判斷「內、外、夾、弓、大、立、腕」的是我們自己建立的特徵、時序分類模型和狀態機。

---

# 建議第一批程式檔

下一步優先建立：

```text
src/
├── hand_detector.py
├── features.py
├── rule_classifier.py
└── realtime.py
```

先把：

```text
Camera
→ MediaPipe
→ Landmark
→ Feature
→ Realtime Display
```

整條 pipeline 跑通。

之後再新增：

```text
collect_data.py
train.py
state_machine.py
```

---

## References

- MediaPipe Hand Landmarker  
  https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

- MediaPipe Gesture Recognizer  
  https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer

- TensorFlow Keras LSTM  
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
