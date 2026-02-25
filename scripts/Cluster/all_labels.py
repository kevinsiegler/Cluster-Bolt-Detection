import os
import shutil
from tqdm import tqdm

def combine_labels():
    # Pfade definieren
    train_dir = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\labels\train"
    val_dir = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\labels\val"
    dest_dir = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\all_labels"

    # Zielordner erstellen
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Ordner erstellt: {dest_dir}")
    else:
        print(f"Ordner existiert bereits: {dest_dir}")

    sources = [train_dir, val_dir]
    total_copied = 0
    
    print("Starte Kopiervorgang...")

    for src in sources:
        if not os.path.exists(src):
            print(f"Warnung: Quellordner nicht gefunden: {src}")
            continue
            
        files = [f for f in os.listdir(src) if f.endswith('.txt')]
        print(f"Kopiere {len(files)} Dateien aus {src}...")
        
        for filename in tqdm(files):
            src_file = os.path.join(src, filename)
            dst_file = os.path.join(dest_dir, filename)
            
            # copy2 behält Metadaten wie Erstellungsdatum bei
            shutil.copy2(src_file, dst_file)
            total_copied += 1

    print(f"Fertig! Insgesamt {total_copied} Dateien kopiert.")
    print(f"Anzahl Dateien im Zielordner: {len(os.listdir(dest_dir))}")

if __name__ == "__main__":
    combine_labels()