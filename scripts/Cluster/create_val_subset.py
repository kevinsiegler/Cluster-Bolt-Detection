r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\create_val_subset.py
"""
import os
import shutil
import random
import yaml
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_subset():
    # Config laden, um Pfade zu erhalten
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Fehler: Config nicht gefunden unter {config_path}")
        return

    cfg = load_config(config_path)
    
    # Quell-Pfade (aus Config oder Annahme basierend auf Context)
    # Da die Config im Context relative Pfade für Output hat, aber absolute für Input:
    val_labels_src = cfg['paths']['val_labels']
    
    # Wir müssen den Pfad zu den Bildern finden. 
    # In der Cluster-Config steht er nicht direkt, aber wir können ihn ableiten 
    # oder wir nehmen an, er liegt parallel zu labels.
    # Basierend auf GNN Config im Context: .../dataset/images/val
    val_images_src = val_labels_src.replace("labels", "images")
    
    if not os.path.exists(val_images_src):
        print(f"Warnung: Bild-Ordner {val_images_src} nicht gefunden. Prüfe Pfad-Struktur.")
        return

    # Ziel-Pfade
    dataset_root = os.path.dirname(os.path.dirname(val_labels_src)) # .../dataset
    subset_images_dst = os.path.join(dataset_root, "images", "val_subset")
    subset_labels_dst = os.path.join(dataset_root, "labels", "val_subset")

    os.makedirs(subset_images_dst, exist_ok=True)
    os.makedirs(subset_labels_dst, exist_ok=True)

    print(f"Quelle Labels: {val_labels_src}")
    print(f"Quelle Bilder: {val_images_src}")
    print(f"Ziel Labels:   {subset_labels_dst}")
    print(f"Ziel Bilder:   {subset_images_dst}")

    # Dateien auflisten (basierend auf Labels, da diese führend sind)
    label_files = [f for f in os.listdir(val_labels_src) if f.endswith('.txt')]
    total_files = len(label_files)
    
    # 10% auswählen
    subset_size = int(total_files * 0.1)
    random.seed(42) # Reproduzierbarkeit
    selected_files = random.sample(label_files, subset_size)
    
    print(f"Erstelle Subset mit {subset_size} von {total_files} Dateien (10%)...")

    for label_file in tqdm(selected_files):
        image_id = os.path.splitext(label_file)[0]
        
        # 1. Label kopieren
        src_lbl = os.path.join(val_labels_src, label_file)
        dst_lbl = os.path.join(subset_labels_dst, label_file)
        shutil.copy2(src_lbl, dst_lbl)
        
        # 2. Bild kopieren (verschiedene Endungen prüfen)
        found_img = False
        for ext in ['.jpg', '.png', '.jpeg']:
            img_name = image_id + ext
            src_img = os.path.join(val_images_src, img_name)
            if os.path.exists(src_img):
                dst_img = os.path.join(subset_images_dst, img_name)
                shutil.copy2(src_img, dst_img)
                found_img = True
                break
        
        if not found_img:
            print(f"Warnung: Kein Bild für ID {image_id} gefunden.")

    print("Subset-Erstellung abgeschlossen.")
    print(f"Neue Pfade für Config (zum Testen):")
    print(f"  val_images: \"{subset_images_dst.replace(os.sep, '/')}\"")
    print(f"  val_labels: \"{subset_labels_dst.replace(os.sep, '/')}\"")

if __name__ == "__main__":
    create_subset()
