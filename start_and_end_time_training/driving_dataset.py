import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class DrivingActionDataset(Dataset):
    def __init__(self, records, num_frames=16, transform=None):
        """
        records: [
          {
            "video_path": "/.../file.mp4",
            "camera": "Dashboard",
            "start_sec": 11.0,
            "end_sec": 31.0,
            "label": 8
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
        end_sec    = rec["end_sec"]
        label      = rec["label"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self.make_black_clip(label)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        if start_frame >= total_frames:
            new_start = max(total_frames - self.num_frames, 0)
            print(f"(UYARI) '{video_path}' için start_frame ({start_frame}) ≥ total_frames ({total_frames}). Çekim start_frame → {new_start}")
            start_frame = new_start
            end_frame = total_frames
        elif end_frame > total_frames:
            print(f"(UYARI) '{video_path}' için end_frame ({end_frame}) > total_frames ({total_frames}). end_frame → {total_frames}")
            end_frame = total_frames

        L = end_frame - start_frame
        frames = []

        if L >= self.num_frames:
            indices = np.linspace(start_frame, end_frame, num=self.num_frames, endpoint=False, dtype=int)
            for fidx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                ret, frame = cap.read()
                if not ret:
                    frames.append(self.black_frame())
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.transform:
                        frame = self.transform(frame)
                    else:
                        frame = frame.astype(np.float32) / 255.0
                        frame = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                    frames.append(frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(min(L, self.num_frames)):
                ret, frame = cap.read()
                if not ret:
                    frames.append(self.black_frame())
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.transform:
                        frame = self.transform(frame)
                    else:
                        frame = frame.astype(np.float32) / 255.0
                        frame = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                    frames.append(frame)
            if len(frames) < self.num_frames:
                if len(frames) > 0:
                    c, h, w = frames[0].shape[0], frames[0].shape[1], frames[0].shape[2]
                else:
                    c, h, w = 3, 112, 112
                for _ in range(self.num_frames - len(frames)):
                    frames.append(torch.zeros((c, h, w), dtype=torch.float32))

        cap.release()

        if len(frames) == 0:
            return self.make_black_clip(label)

        clip = torch.stack(frames, dim=1)
        return clip, label

    def black_frame(self):
        C = 3
        H, W = 112, 112
        return torch.zeros((C, H, W), dtype=torch.float32)

    def make_black_clip(self, label):
        C = 3
        T = self.num_frames
        H, W = 112, 112
        clip = torch.zeros((C, T, H, W), dtype=torch.float32)
        return clip, label
