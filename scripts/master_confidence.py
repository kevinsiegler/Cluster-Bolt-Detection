# scripts/master_confidence.py
import subprocess
import sys

# Confidence-Werte: 0.2 → 1.0 in 0.05 Schritten
conf_values = [round(0.25 + 0.05 * i, 2) for i in range(4)]

python_executable = sys.executable  # garantiert gleiche Python-Version

for conf in conf_values:
    print(f"\n🚀 Starte neuen Prozess für conf={conf}")

    subprocess.run(
        [
            python_executable,
            "confidence_analysis.py",
            str(conf)
        ],
        check=True
    )

    print("🧹 Prozess beendet – Speicher vollständig freigegeben")

print("\n🎉 Alle Confidence-Durchläufe abgeschlossen.")
