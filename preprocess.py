import os
import glob
import pickle
import pandas as pd


PROJECT_ROOT = "/Users/erinc/GitHub/NaturalisticDrivingActionRecognition"

DATA_ROOT = os.path.expanduser("/Users/erinc/Desktop/Computer Vision Dataset")

FOLDER = "A1_1"

EXT = ".mp4"
# -----------------------------------------------------

def time_to_sec(t: str) -> float:
    """HH:MM:SS → toplam saniye (float)."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

def parse_annotations(csv_path: str, video_root: str, ext: str = ".mp4"):
    """
    CSV dosyasını okuyup her satırı şu dict formatına çevirir:
      {
        "video_path": tam dosya yolu,
        "camera": "Dashboard"/"Rearview"/"Rightside Window",
        "start_sec": float (başlangıç saniyesi),
        "end_sec": float (bitiş saniyesi),
        "label": int (0–15)
      }
    """
    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        filename = row["Filename"] + ext
        camera   = row["Camera View"]
        label_id = int(row["Label (Primary)"].replace("Class ", "").strip())

        start_sec = time_to_sec(row["Start Time"])
        end_sec   = time_to_sec(row["End Time"])

        video_path = os.path.join(video_root, filename)
        if not os.path.exists(video_path):
            print(f"UYARI: Video bulunamadı: {video_path}")
            continue

        records.append({
            "video_path": video_path,
            "camera": camera,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label_id
        })

    return records

def process_one_folder(data_root: str, project_root: str, folder_name: str, ext: str = ".mp4"):
    """
    data_root: Masaüstündeki veri dizini (örneğin "~/Desktop/DrivingData").
    project_root: Python dosyalarınızı tuttuğunuz proje klasörü.
    a1_folder_name: "A1_1", "A1_2", ... vb.
    ext: Video uzantısı (".mp4" veya ".avi").

    1) data_root/A1_k içindeki tüm user_id_* klasöründen CSV'leri parse eder.
    2) Bir liste (all_records) olarak toplar.
    3) Bu listeyi project_root/records_A1_k.pkl olarak kaydeder.
    """
    folder_path = os.path.join(data_root, folder_name)
    if not os.path.isdir(folder_path):
        print(f"HATA: '{folder_path}' dizini bulunamadı. Lütfen DATA_ROOT ve A1_FOLDER değerlerini kontrol edin.")
        return

    print(f"\n>>> İşlem Başlıyor: {folder_path}")

    all_records = []
    user_dirs = glob.glob(os.path.join(folder_path, "user_id_*"))
    if len(user_dirs) == 0:
        print(f"UYARI: {folder_path} içinde hiçbir 'user_id_*' klasörü bulunamadı.")
    for user_dir in user_dirs:
        csv_files = glob.glob(os.path.join(user_dir, "*.csv"))
        if len(csv_files) == 0:
            print(f"UYARI: {user_dir} içinde CSV bulunamadı, atlanıyor.")
            continue

        for csv_path in csv_files:
            recs = parse_annotations(csv_path, user_dir, ext=ext)
            all_records.extend(recs)

    if len(all_records) == 0:
        print(f"UYARI: '{folder_name}' içinde tek bir geçerli segment dahi bulunamadı. Dosya oluşturulmuyor.")
        return

    out_pkl = os.path.join(project_root, f"records_{folder_name}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(all_records, f)
    print(f">>> '{out_pkl}' oluşturuldu. Toplam segment sayısı: {len(all_records)}")


if __name__ == "__main__":
    process_one_folder(DATA_ROOT, PROJECT_ROOT, FOLDER, ext=EXT)
