import cv2
import torch
import torchvision.transforms as T
from torchvision.models.video import r3d_18

# 1.1. Modeli tanımlayın (aynı train_full.py’daki gibi)
def get_video_model(num_classes=16, pretrained=False):
    model = r3d_18(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = torch.nn.Linear(in_feat, num_classes)
    return model

# Cihaz seçimi
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
checkpoint_path = "final_best_model_checkpoint_with_end_time.pth"   # kendi projenizdeki tam yol
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 2.1.1. Transform tanımı (train sırasında kullandığınızla aynı olmalı)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((112,112)),
    T.ToTensor(),
    T.Normalize(mean=[0.43216,0.394666,0.37645],
                std =[0.22803,0.22145,0.216989])
])

# 2.1.2. Etiketsiz yeni video yolu ve çıktı video yolu
video_path = r"D:\Computer Vision Dataset\A1_3\user_id_41850\Rearview_user_id_41850_5.mp4"
output_path = "example_video_1.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Videoyu açılamadı: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Çözünürlük: {width}x{height}, Toplam Frame: {total_frames}")

# VideoWriter ayarları
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 2.1.3. Parametreler
num_frames = 16
stride = 8
CONF_THRESHOLD = 0.7
labels_map = {
    0: "Normal Forward Driving",
    1: "Drinking",
    2: "Phone Call(right)",
    3: "Phone Call(left)",
    4: "Eating",
    5: "Text (Right)",
    6: "Text (Left)",
    7: "Reaching behind",
    8: "Adjust control panel",
    9: "Pick up from floor (Driver)",
    10: "Pick up from floor (Passenger)",
    11: "Talk to passenger at the right",
    12: "Talk to passenger at backseat",
    13: "Yawning",
    14: "Hand on head",
    15: "Singing or dancing with music"
}

# 2.1.4. Sliding window ile tahmin yapıp sonuçları kaydet
results = []  # her klip için dict: start_frame, end_frame, label_name, confidence
frame_idx = 0
while frame_idx + num_frames <= total_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = transform(rgb)
        frames.append(rgb)
    if len(frames) < num_frames:
        break
    clip_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(clip_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_label = pred.item()
        pred_conf  = conf.item()
    start_frame = frame_idx
    end_frame = frame_idx + num_frames
    if pred_conf >= CONF_THRESHOLD:
        results.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "label_name": labels_map[pred_label],
            "confidence": pred_conf
        })
    frame_idx += stride

# 2.1.5. Frame bazında anotasyon ekleyip yeni video yaz
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
current_frame = 0
res_idx = 0
active = False
active_end = -1
active_label = ""
active_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Eğer yeni segment aktifleşecekse başlat
    if res_idx < len(results) and current_frame == results[res_idx]["start_frame"]:
        active = True
        active_end = results[res_idx]["end_frame"]
        active_label = results[res_idx]["label_name"]
        active_conf = results[res_idx]["confidence"]
        res_idx += 1

    # Eğer segment aktifse ve henüz bitişe ulaşılmadıysa, anotasyonu çiz
    if active and current_frame < active_end:
        # Dikdörtgen (frame kenarına)
        cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 4)
        # Metin: sınıf ve confidence
        text = f"{active_label}: {active_conf:.2f}"
        cv2.rectangle(frame, (50, 20), (50 + len(text)*12, 50), (0, 255, 0), -1)
        cv2.putText(frame, text, (55, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    # Eğer segment bitti ise pasif duruma geç
    if active and current_frame >= active_end:
        active = False

    out.write(frame)
    current_frame += 1

cap.release()
out.release()
