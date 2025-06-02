import torch
import torchvision.transforms as T
from torchvision.models.video import r3d_18

# 1.1. Modeli tanımlayın (aynı train_full.py’daki gibi)
def get_video_model(num_classes=16, pretrained=False):
    model = r3d_18(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = torch.nn.Linear(in_feat, num_classes)
    return model

# 1.2. Cihazı (device) seçin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Inference için device:", device)

# 1.3. Modeli oluşturun ve eğitilmiş ağırlıkları yükleyin
model = get_video_model(num_classes=16, pretrained=False)
checkpoint_path = "checkpoint_full.pth"   # kendi projenizdeki tam yol
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

import cv2
import torch
import numpy as np

# 2.1.1. Transform tanımı (train sırasında kullandığınızla aynı olmalı)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((112,112)),
    T.ToTensor(),
    T.Normalize(mean=[0.43216,0.394666,0.37645],
                std =[0.22803,0.22145,0.216989])
])

# 2.1.2. Etiketsiz yeni video yolu
video_path = "/path/to/yeni_etiketsiz_video.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Videoyu açılamadı: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Toplam Frame: {total_frames}")

# 2.1.3. Parametreler
num_frames = 16       # modelde kullandığınız klip uzunluğu
stride = 8            # kaç frame kaydırayım? (overlap=50% için 8 önerilir)
labels_map = {
    0: "Eating", 1: "Yawning", 2: "Smoking", 3: "Phone Use",
    4: "Talking to Passenger", 5: "Reaching Back", 6: "Hair/Makeup",
    7: "Drinking", 8: "Searching", 9: "Adjusting Radio", 10: "Adjusting Mirrors",
    11: "Using Controls", 12: "Distracted with Hat", 13: "Hands Off Wheel",
    14: "Other", 15: "Unlabeled"
}

# 2.1.4. Sliding window ile kare toplayıp tahmin al
results = []  # her klip için (start_frame, end_frame, tahmin_label, tahmin_confidence)

frame_idx = 0
while frame_idx + num_frames <= total_frames:
    # 2.1.4.1. Klip için gerekli kareleri oku
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)         # (3, H, W) normalized
        frames.append(frame)

    if len(frames) < num_frames:
        # Video sonuna yakınken tam kare alınamadıysa bitir
        break

    # 2.1.4.2. Klipi (3, num_frames, H, W) şekline getir
    clip_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    # (1, 3, num_frames, H, W)

    # 2.1.4.3. Model ile tahmin
    with torch.no_grad():
        outputs = model(clip_tensor)   # (1, 16) → logits
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_label = pred.item()       # 0–15 arası
        pred_conf  = conf.item()       # olasılık

    start_time = frame_idx / fps
    end_time   = (frame_idx + num_frames) / fps
    results.append({
        "start_sec": start_time,
        "end_sec": end_time,
        "label_id": pred_label,
        "label_name": labels_map[pred_label],
        "confidence": pred_conf
    })

    frame_idx += stride

cap.release()

# 2.1.5. Sonuçları yazdır
for r in results:
    print(f"{r['start_sec']:.2f}s–{r['end_sec']:.2f}s → "
          f"{r['label_id']} ({r['label_name']}), conf={r['confidence']:.3f}")

import cv2
import torch
import numpy as np

# 1. Video’yu aç
video_path = "/path/to/etiketsiz_video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

# 2. Her 1 saniyede bir klip seç
interval_secs = 1.0
num_samples = int(video_duration // interval_secs)

pred_counts = np.zeros(16, dtype=int)  # her sınıf için sayaç

for i in range(num_samples):
    t = i * interval_secs
    start_frame = int(t * fps)
    end_frame = start_frame + 16  # 16 karelik pencere

    if end_frame > total_frames:
        break

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(16):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    if len(frames) < 16:
        break

    clip = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(clip)
        pred = torch.argmax(outputs, dim=1).item()
    pred_counts[pred] += 1

cap.release()

# 3. Videonun genel etiketi: en sık çıkan sınıf
final_label = np.argmax(pred_counts)
print("Video genel tahmin:", final_label, "(", labels_map[final_label], ")")
