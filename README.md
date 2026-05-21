# PostureDetection

Smartphone IMU 기반 자세 분류 프로젝트입니다. Android 앱으로 accelerometer와 gyroscope 데이터를 수집하고, 수집한 CSV 데이터로 1D-CNN 모델을 학습한 뒤, 별도의 Android 앱에서 TensorFlow Lite 모델로 실시간 자세를 감지합니다.

분류 대상 자세는 다음 3개입니다.

- `Upright`
- `Mild Forward Flexion`
- `Severe Looking-Down`

## Project Structure

```text
PostureDetection/
├── IMUDataCollectorApp/          # IMU CSV 수집용 Android 앱
├── PostureIMUData+Training/      # 수집 CSV, 학습 스크립트, 학습된 모델
└── PostureDetectionApp/          # 실시간 자세 감지 Android 앱
```

## IMUDataCollectorApp

`IMUDataCollectorApp`은 학습 데이터를 만들기 위한 Android 앱입니다.

주요 기능:

- accelerometer와 gyroscope 수집
- `Upright`, `Mild Forward Flexion`, `Severe Looking-Down` label 선택
- `Sitting`, `Standing`, `Commuting-like`, `Resting` context 기록
- 30초 calibration 지원
- CSV 파일 저장

저장되는 CSV 주요 컬럼:

```text
timestamp_ns, system_time_ms, session_id, day_id, label, context,
calibrated, baseline_ax, baseline_ay, baseline_az,
ax, ay, az, gx, gy, gz, screen_rotation, note
```

## PostureIMUData+Training

`PostureIMUData+Training`에는 총 16개의 `posture_session_*.csv` 데이터와 학습 스크립트가 있습니다.

학습 입력:

```text
window shape = (100, 6)
100 = 50Hz 기준 2초
6 = ax, ay, az, gx, gy, gz
```

Window 생성 방식:

- window size: `100`
- hop size: `50`
- 같은 label segment 안에서만 window 생성
- transition margin 기본값: `0.25초`
- 50Hz 기준 segment 앞/뒤 `12 rows` 제거 후 window 생성
- source file 기준 train/validation/test split

기본 split:

```text
train: 앞 12개 CSV 파일
validation: 다음 2개 CSV 파일
test: 마지막 2개 CSV 파일
```

### Training

Python 3.11 가상환경 사용을 권장합니다. 이 프로젝트에는 로컬 실행용 `.venv`가 만들어져 있습니다.

```bash
cd "PostureIMUData+Training"
source .venv/bin/activate
python train_posture_1dcnn.py
```

옵션:

```bash
python train_posture_1dcnn.py \
  --data_dir . \
  --model_dir models \
  --epochs 50 \
  --transition_margin_sec 0.25
```

필요 패키지:

```bash
pip install numpy pandas scikit-learn tensorflow
```

### Model

학습 모델은 TensorFlow/Keras 1D-CNN입니다.

```text
Input(shape=(100, 6))
Conv1D(32, kernel_size=5)
BatchNormalization
Conv1D(64, kernel_size=5)
BatchNormalization
MaxPooling1D
Conv1D(128, kernel_size=3)
BatchNormalization
GlobalAveragePooling1D
Dense(64)
Dropout(0.3)
Dense(3, softmax)
```

생성되는 주요 파일:

```text
PostureIMUData+Training/models/best_posture_1dcnn.keras
PostureIMUData+Training/models/posture_1dcnn_final.keras
PostureIMUData+Training/models/normalization.json
PostureIMUData+Training/models/training_history.csv
PostureIMUData+Training/models/confusion_matrix.csv
PostureIMUData+Training/models/classification_report.txt
PostureIMUData+Training/models/posture_1dcnn_float32.tflite
PostureIMUData+Training/models/posture_1dcnn_int8.tflite
```

현재 `models` 기준 test 결과:

```text
Test accuracy: 0.9290

Recall:
Upright: 0.8588
Mild Forward Flexion: 0.9363
Severe Looking-Down: 0.9922
```

## PostureDetectionApp

`PostureDetectionApp`은 학습된 TFLite 모델을 이용해 실시간 자세를 감지하는 Android 앱입니다.

주요 기능:

- foreground service 기반 background posture monitoring
- accelerometer와 gyroscope를 50Hz로 sampling
- 최근 100개 sample을 ring buffer에 저장
- 1초마다 TFLite inference 실행
- 3개 prediction moving average smoothing
- INT8 모델 기본 사용, FLOAT32 모델 fallback 지원
- 나쁜 자세가 8초 이상 지속되면 알림 및 진동 경고

앱에서 사용하는 asset:

```text
PostureDetectionApp/app/src/main/assets/posture_1dcnn_int8.tflite
PostureDetectionApp/app/src/main/assets/posture_1dcnn_float32.tflite
PostureDetectionApp/app/src/main/assets/normalization.json
```

학습 후 앱 모델을 갱신하려면:

```bash
cp "PostureIMUData+Training/models/posture_1dcnn_int8.tflite" \
   "PostureDetectionApp/app/src/main/assets/posture_1dcnn_int8.tflite"

cp "PostureIMUData+Training/models/posture_1dcnn_float32.tflite" \
   "PostureDetectionApp/app/src/main/assets/posture_1dcnn_float32.tflite"

cp "PostureIMUData+Training/models/normalization.json" \
   "PostureDetectionApp/app/src/main/assets/normalization.json"
```

## Build

각 Android 앱은 독립된 Gradle 프로젝트입니다.

데이터 수집 앱:

```bash
cd IMUDataCollectorApp
./gradlew assembleDebug
```

자세 감지 앱:

```bash
cd PostureDetectionApp
./gradlew assembleDebug
```

## Notes
- `PostureIMUData+Training/models/`가 실제 사용 모델 폴더입니다.
- `models_smoke_test`, `models_margin_smoke_test`, `models_margin_025`는 실험/검증용 출력 폴더입니다.
- Python 3.14에서는 TensorFlow wheel이 제공되지 않을 수 있으므로 Python 3.11 사용을 권장합니다.
