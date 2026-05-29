import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# ==========================================
# 0. 萬惡環境防禦補丁 (防止 TensorFlow 崩潰與找不到版本)
# ==========================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# 強制關閉 GPU 呼叫，用 CPU 跑最穩定
tf.config.set_visible_devices([], 'GPU')

if not hasattr(tf, "__version__"):
    tf.__version__ = "2.12.0"

from deepface import DeepFace

# ==========================================
# 1. 設定路徑 (根據你的 Kaggle 解壓結構)
# ==========================================
source_base_dir = Path("data")
emotions = ["Angry", "Happy", "Sad"]

# 目標資料夾
face_data_dir = Path("face_data")
face_data_ok_dir = Path("face_data_ok")

# 建立需要的資料夾
face_data_dir.mkdir(parents=True, exist_ok=True)
face_data_ok_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. 步驟一：整合所有圖片至 face_data 資料夾
# ==========================================
print("【步驟一】正在整合圖片至 face_data 資料夾...")

file_count = 0
for emotion in emotions:
    emotion_dir = source_base_dir / emotion

    if not emotion_dir.exists():
        print(f"警告: 找不到資料夾 {emotion_dir}，跳過此類別。")
        continue

    # 支援常見的影像格式
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(emotion_dir.glob(ext))

    for img_path in tqdm(image_paths, desc=f"複製 {emotion} 圖片"):
        # 加上情緒字首避免同名覆蓋
        new_filename = f"{emotion}_{img_path.name}"
        dest_path = face_data_dir / new_filename

        shutil.copy(img_path, dest_path)
        file_count += 1

print(f"總共複製了 {file_count} 張圖片至 {face_data_dir}/\n")

# ==========================================
# 3. 步驟二：利用 OpenCV 抓取人臉，再用 DeepFace 分析表情並畫框
# ==========================================
print("【步驟二】正在使用 OpenCV + DeepFace 進行人臉表情分析與畫框...")

# 載入 OpenCV 內建的人臉特徵模型 (對接你提供的小程式邏輯)
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

all_faces = list(face_data_dir.glob("*"))
success_count = 0
fail_count = 0

for img_path in tqdm(all_faces, desc="DeepFace 表情分析中"):
    if not img_path.is_file():
        continue
    if img_path.name.startswith("."):
        continue

    try:
        # 使用安全讀取法（避免 Windows 中文/特殊路徑死掉）
        img_array = np.fromfile(str(img_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            fail_count += 1
            continue

        # 轉成灰階圖供 Haar Cascade 偵測人臉
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            fail_count += 1
            continue

        has_analyzed_face = False

        # 走訪這張圖偵測到的每一張臉
        for (x, y, w, h) in faces:
            # 裁剪出人臉區域給 DeepFace 分析
            face_img = img[y:y+h, x:x+w]

            try:
                # 呼叫 DeepFace 進行表情分析 (detector_backend='skip' 沿用你的設定速度最快)
                result = DeepFace.analyze(
                    img_path=face_img,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip"
                )

                # 拿到主要情緒標籤
                dominant_emotion = result[0]["dominant_emotion"]

                # 在原圖上畫綠色框 (0, 255, 0)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # 在框上方寫上情緒文字
                cv2.putText(
                    img,
                    dominant_emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # 字體大小稍微縮小一點防止超框
                    (0, 255, 0),
                    2
                )
                has_analyzed_face = True

            except Exception as e:
                # 某張臉分析失敗就跳過
                continue

        if has_analyzed_face:
            # 設定輸出路徑到 face_data_ok 資料夾
            output_path = face_data_ok_dir / img_path.name

            # 安全寫入檔案 (支援 Windows 編碼環境)
            is_success, im_buf = cv2.imencode(img_path.suffix, img)
            if is_success:
                with open(output_path, "wb") as f:
                    f.write(im_buf)
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1

    except Exception as e:
        fail_count += 1

print("\n==========================================")
print("任務完成統計：")
print(f"成功分析表情並畫框的圖片數 (face_data_ok): {success_count} 張")
print(f"未能辨識或分析失敗的圖片數: {fail_count} 張")
print("==========================================")