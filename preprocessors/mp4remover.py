import os

def remove_noaudio_from_filenames(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        print(f"📂 Klasör: {foldername}")
        for filename in filenames:
            print(f"  🔎 Dosya bulundu: {filename}")
            if "NoAudio_" in filename:
                new_filename = filename.replace("NoAudio_", "")
                old_path = os.path.join(foldername, filename)
                new_path = os.path.join(foldername, new_filename)

                print(f"  🛠 Yeniden adlandırılıyor: {old_path} -> {new_path}")
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  ✅ Başarılı: {filename} -> {new_filename}")
                else:
                    print(f"  ⚠️ Zaten var: {new_filename}")

# Buraya klasör yolunu KESİN DOĞRU gir!
remove_noaudio_from_filenames(r"C:\Users\erinc\Desktop\Computer Vision Dataset\A1_7")

