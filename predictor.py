import torch
import torchvision.transforms as T
from torchvision.models.video import r3d_18

# 1.1. Modeli tanımlayın (aynı train_full.py’daki gibi)
def get_video_model(num_classes=16, pretrained=False):
    model = r3d_18(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = torch.nn.Linear(in_feat, num_classes)
    return model

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU kullanılacak.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Mac GPU) kullanılacak.")
else:
    device = torch.device("cpu")
    print("CPU kullanılacak.")

# 1.3. Modeli oluşturun ve eğitilmiş ağırlıkları yükleyin
model = get_video_model(num_classes=16, pretrained=False)
checkpoint_path = "path to checkpoint"   # kendi projenizdeki tam yol
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

import cv2
import torch

# 2.1.1. Transform tanımı (train sırasında kullandığınızla aynı olmalı)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((112,112)),
    T.ToTensor(),
    T.Normalize(mean=[0.43216,0.394666,0.37645],
                std =[0.22803,0.22145,0.216989])
])

# 2.1.2. Etiketsiz yeni video yolu
video_path = "path to video"

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
        outputs = model(clip_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_label = pred.item()
        pred_conf  = conf.item()

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
