import os
import glob
import pickle
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.video import r3d_18
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from driving_dataset import DrivingActionDataset  # Yeni dataset sınıfı

# -------------------- DÜZENLEYİN --------------------
# Proje klasörünüz: bu script'in bulunduğu dizin
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Hiperparametreler:
BATCH_SIZE_FULL     = 8      # batch size
EPOCHS_FULL         = 20     # kaç epoch
LR_FULL             = 1e-4   # öğrenme hızı
NUM_WORKERS_FULL    = 4      # DataLoader için
# -----------------------------------------------------

def filter_unreadable_records(records):
    """
    records: [ {"video_path","start_sec",...}, ... ]
    Her record için başlangıç frame’inden en az bir kare okunabiliyorsa kayda devam, yoksa at.
    """
    good = []
    for rec in records:
        video_path = rec["video_path"]
        start_sec  = rec["start_sec"]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, _ = cap.read()
        cap.release()
        if ret:
            good.append(rec)
    return good

def get_video_model(num_classes=16, pretrained=True):
    model = r3d_18(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for clips, labels in tqdm(dataloader, desc="  ● Eğitim   "):
        clips = clips.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += clips.size(0)

    return total_loss / total_count, total_correct / total_count

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for clips, labels in tqdm(dataloader, desc="  ● Doğrulama   "):
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * clips.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += clips.size(0)

    return total_loss / total_count, total_correct / total_count

if __name__ == "__main__":
    # ---------------- Device seçimi ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU kullanılacak.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Mac GPU) kullanılacak.")
    else:
        device = torch.device("cpu")
        print("CPU kullanılacak.")

    # 1) Proje klasörünüzdeki tüm 'records_A1_*.pkl' dosyalarını bul
    pattern = os.path.join(PROJECT_ROOT, "records_A1_*.pkl")
    pkl_files = sorted(glob.glob(pattern))
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"'{pattern}' ile eşleşen hiçbir dosya bulunamadı. "
                                "Önce process_folder.py ile pickle oluşturun ve kontrol edin.")

    all_records = []
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            recs = pickle.load(f)
        print(f"  '{os.path.basename(pkl_path)}' → {len(recs)} segment (ham)")
        # 1a) Her pickle'dan gelen kayıtları filtreleyip bozuklarını atalım:
        recs_filtered = filter_unreadable_records(recs)
        print(f"      Filtre sonrası: {len(recs_filtered)} kayıt")
        all_records.extend(recs_filtered)

    print(f"\n>>> Toplam segment sayısı (tüm klasörlerden, filtre sonrası): {len(all_records)}\n")

    # 2) Train/Val split (%80-%20 stratified)
    labels = [r["label"] for r in all_records]
    train_recs, val_recs = train_test_split(
        all_records,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"  Train: {len(train_recs)}  |  Val: {len(val_recs)}\n")

    # 3) Dataset & DataLoader
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112,112)),
        T.ToTensor(),
        T.Normalize(mean=[0.43216,0.394666,0.37645], std=[0.22803,0.22145,0.216989])
    ])
    train_ds = DrivingActionDataset(train_recs, num_frames=16, transform=transform)
    val_ds   = DrivingActionDataset(val_recs,   num_frames=16, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_FULL,
        shuffle=True,
        num_workers=NUM_WORKERS_FULL,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_FULL,
        shuffle=False,
        num_workers=NUM_WORKERS_FULL,
        pin_memory=True
    )

    # 4) Model, Kayıp, Optimizasyon
    model = get_video_model(num_classes=16, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-5)

    # 5) Eğitim Döngüsü (EPOCHS_FULL kadar)
    best_val_acc = 0.0
    checkpoint_full = os.path.join(PROJECT_ROOT, "checkpoint_full.pth")

    for epoch in range(1, EPOCHS_FULL + 1):
        print(f"\nEpoch {epoch}/{EPOCHS_FULL}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Eğitim:    Loss={train_loss:.4f}, Acc={train_acc:.4f}")

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"  Doğrulama: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_full)
            print(f"  >> Yeni en iyi model: {checkpoint_full}")

    print(f"\n>>> Tam ölçekli eğitim tamamlandı.")
    print(f">>> En iyi val doğruluğu: {best_val_acc:.4f}")
    print(f">>> Final modeli: {checkpoint_full}")
