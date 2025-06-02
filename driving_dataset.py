import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class DrivingActionDataset(Dataset):
    def __init__(self, records, num_frames=16, transform=None):
        """
        records: [
          {
            "video_path": "/.../Right_side_window_user_id_16700_7.mp4",
            "camera": "Rightside Window",
            "start_sec": 484.0,
            "end_sec": 507.0,
            "label": 14
          },
          ...
        ]
        num_frames: Her clip'te kaç kare alınacağı (örneğin 16).
        transform: Her kareye uygulanacak torchvision transform.
        """
        self.records = records
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        video_path = rec["video_path"]
        start_sec  = rec["start_sec"]
        label      = rec["label"]

        # VideoCapture açılıyor
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Video açılamadıysa siyah clip + etiket döndür
            return self.make_black_clip(label)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_sec * fps)

        # Eğer segment video süresini aşıyorsa, son num_frames kareden başla
        if start_frame >= total_frames:
            new_start = total_frames - self.num_frames
            if new_start < 0:
                new_start = 0
            print(f"(UYARI) '{video_path}' için start_frame ({start_frame}) total_frames ({total_frames}) aşıyor. "
                  f"start_frame --> {new_start}")
            start_frame = new_start

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = frame.astype(np.float32) / 255.0
                frame = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            frames.append(frame)

        cap.release()

        # Eğer hiç kare okunamadıysa (örneğin bozuk video)
        if len(frames) == 0:
            return self.make_black_clip(label)

        # Eksik kare ise siyah doldurma
        if len(frames) < self.num_frames:
            c, h, w = frames[0].shape[0], frames[0].shape[1], frames[0].shape[2]
            for _ in range(self.num_frames - len(frames)):
                frames.append(torch.zeros((c, h, w), dtype=torch.float32))

        clip = torch.stack(frames, dim=1)  # (3, num_frames, H, W)
        return clip, label

    def make_black_clip(self, label):
        """
        Bozuk video veya kare okunmazsa buraya düşer.
        Siyah (zeros) bir clip üretip etiketle döndürür.
        """
        C = 3
        T = self.num_frames
        # Eğer transform’ta Resize((112,112)) kullanıyorsanız:
        H, W = 112, 112

        clip = torch.zeros((C, T, H, W), dtype=torch.float32)
        return clip, label
