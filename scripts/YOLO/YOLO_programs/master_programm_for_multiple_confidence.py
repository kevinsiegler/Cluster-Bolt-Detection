import subprocess
import sys
import os

python_executable = sys.executable

# Ordner mit ALLEN Trainingsläufen
TRAIN_BASE_DIR = r"C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\runs\detect\train_w_different_amounts_data"

# Alle Unterordner einsammeln, die mit "train_" beginnen
train_runs = sorted([
    d for d in os.listdir(TRAIN_BASE_DIR)
    if d.startswith("train_") and
       os.path.isdir(os.path.join(TRAIN_BASE_DIR, d))
])

print(f"🔍 Gefundene Trainingsmodelle: {len(train_runs)}")

for train_name in train_runs:
    print(f"\n🚀 Starte Inferenz für Modell: {train_name}")

    subprocess.run(
        [
            python_executable,
            "sub_programm_for_multiple_confidence.py",
            train_name
        ],
        check=True
    )

    print("🧹 Modell abgeschlossen – Speicher freigegeben")

print("\n🎉 Alle Trainingsmodelle wurden ausgewertet.")
