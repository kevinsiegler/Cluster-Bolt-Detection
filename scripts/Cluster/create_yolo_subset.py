import os
import shutil
import random
import yaml
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_subset():
    # Pfad zur Config-Datei ermitteln
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Fehler: Config nicht gefunden unter {config_path}")
        return

    cfg = load_config(config_path)
    
    # Den aktuellen Input-Ordner aus der Config lesen
    input_dir = cfg['paths']['inference_input_dir']
    
    # Falls die Config bereits auf ein Subset zeigt (Sicherheitscheck), versuchen wir das Original zu finden
    if input_dir.endswith("_subset"):
        print("Info: Der Pfad in der Config scheint bereits ein Subset zu sein.")
        print("Versuche, den ursprünglichen Ordner abzuleiten...")
        input_dir = input_dir.replace("_subset", "")
    
    if not os.path.exists(input_dir):
        print(f"Fehler: Quell-Ordner {input_dir} nicht gefunden.")
        return

    # Ziel-Ordner definieren (gleicher Pfad + "_subset")
    # rstrip, um eventuelle abschließende Slashes zu entfernen, damit der Suffix korrekt angehängt wird
    subset_dir = input_dir.rstrip("/\\") + "_subset"

    # Falls der Ordner schon existiert, vorher bereinigen, um ein sauberes 5% Sample zu haben
    if os.path.exists(subset_dir):
        print(f"Bereinige alten Subset-Ordner: {subset_dir}")
        shutil.rmtree(subset_dir)
    
    os.makedirs(subset_dir)

    print(f"Quelle: {input_dir}")
    print(f"Ziel:   {subset_dir}")

    # Dateien auflisten
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    total_files = len(files)
    
    # 5% berechnen
    subset_size = int(total_files * 0.10)
    if subset_size < 1 and total_files > 0:
        subset_size = 1
        
    random.seed(42) # Für Reproduzierbarkeit
    selected_files = random.sample(files, subset_size)
    
    print(f"Erstelle 5% Subset ({subset_size} von {total_files} Dateien)...")
    
    for f in tqdm(selected_files):
        src = os.path.join(input_dir, f)
        dst = os.path.join(subset_dir, f)
        shutil.copy2(src, dst)
        
    print("Fertig!")
    print(f"Der neue Pfad für die Config ist:\n{subset_dir}")

if __name__ == "__main__":
    create_subset()
