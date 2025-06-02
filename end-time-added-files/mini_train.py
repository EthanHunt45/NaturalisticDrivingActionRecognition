import os
import pickle
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.video import r3d_18
from tqdm import tqdm

from driving_dataset import DrivingActionDataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDS_PKL_NAME = "records_A1_1.pkl"
BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-4
NUM_WORKERS = 2

def filter_unreadable_records(records):
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    records_pkl_path = os.path.join(PROJECT_ROOT, RECORDS_PKL_NAME)
    with open(records_pkl_path, "rb") as f:
        all_records = pickle.load(f)
    print(f">>> Yüklenen toplam segment sayısı (ham): {len(all_records)}")

    filtered_records = filter_unreadable_records(all_records)
    print(f">>> Filtre sonrası segment sayısı: {len(filtered_records)}")

    random.shuffle(filtered_records)
    split_idx = int(0.8 * len(filtered_records))
    train_recs = filtered_records[:split_idx]
    val_recs   = filtered_records[split_idx:]
    print(f"  Train: {len(train_recs)}  |  Val: {len(val_recs)}")

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112,112)),
        T.ToTensor(),
        T.Normalize(mean=[0.43216,0.394666,0.37645], std=[0.22803,0.22145,0.216989])
    ])
    train_ds = DrivingActionDataset(train_recs, num_frames=16, transform=transform)
    val_ds   = DrivingActionDataset(val_recs,   num_frames=16, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = get_video_model(num_classes=16, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    best_val_acc = 0.0
    base_name = RECORDS_PKL_NAME.replace("records_", "").replace(".pkl", "")
    checkpoint_name = os.path.join(PROJECT_ROOT, f"checkpoint_{base_name}_small.pth")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Eğitim:    Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"  Doğrulama: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  >> Yeni en iyi model: {checkpoint_name}")

    print(f"\n>>> Küçük ölçekli eğitim tamamlandı. En iyi val doğruluğu: {best_val_acc:.4f}")
    print(f">>> Ağırlık dosyası: {checkpoint_name}")
