# scripts/master_confidence.py
import subprocess
import sys

# Confidence-Werte: 0.2 → 1.0 in 0.05 Schritten
conf_values = [round(0.05 + 0.05 * i, 2) for i in range(19)]

python_executable = sys.executable  # garantiert gleiche Python-Version

for conf in conf_values:
    print(f"\n🚀 Starte neuen Prozess für conf={conf}")

    subprocess.run(
        [
            python_executable,
            "sub_programm_for_multiple_confidence.py",
            str(conf)
        ],
        check=True
    )

    print("🧹 Prozess beendet – Speicher vollständig freigegeben")

print("\n🎉 Alle Confidence-Durchläufe abgeschlossen.")
