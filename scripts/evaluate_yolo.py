import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def load_labels(label_dir):
    labels = {}
    for path in glob(os.path.join(label_dir, "*.txt")):
        name = os.path.basename(path)
        with open(path, "r") as f:
            data = []
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    data.append(parts)  # [class, x, y, w, h]
            labels[name] = np.array(data)
    return labels

def bbox_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def evaluate_model(gt_dir, pred_dir, iou_threshold=0.5):
    gt_labels = load_labels(gt_dir)
    pred_labels = load_labels(pred_dir)
    stats = []

    for img_name, gt_boxes in gt_labels.items():
        pred_boxes = pred_labels.get(img_name, np.array([]))
        matched_pred = set()

        for gt_box in gt_boxes:
            gt_class = int(gt_box[0])
            best_iou = 0
            best_pred = None

            for i, pred_box in enumerate(pred_boxes):
                iou = bbox_iou(gt_box[1:], pred_box[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = (i, pred_box)

            if best_iou >= iou_threshold and best_pred is not None:
                pred_class = int(best_pred[1][0])
                matched_pred.add(best_pred[0])
                correct = gt_class == pred_class
                stats.append({
                    "image": img_name,
                    "iou": best_iou,
                    "gt_class": gt_class,
                    "pred_class": pred_class,
                    "match": correct
                })
            else:
                stats.append({
                    "image": img_name,
                    "iou": 0,
                    "gt_class": gt_class,
                    "pred_class": None,
                    "match": False
                })

        for j, pred_box in enumerate(pred_boxes):
            if j not in matched_pred:
                stats.append({
                    "image": img_name,
                    "iou": 0,
                    "gt_class": None,
                    "pred_class": int(pred_box[0]),
                    "match": False
                })
    df = pd.DataFrame(stats)
    return df

# ============================================================
# DASHBOARD
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_GT = os.path.join(BASE_DIR, "dataset", "labels", "val")
DEFAULT_PRED = os.path.join(BASE_DIR, "scripts", "runs", "detect", "predict", "labels")

st.title("🔩 Vollumfängliches YOLOv8 Dashboard – Schraubenanalyse")
st.write("Dieses Interface bewertet die Leistung deiner KI bei der Erkennung von Schrauben und fehlenden Schrauben.")
st.write(f"Basisverzeichnis: `{BASE_DIR}`")

gt_dir = st.text_input("📁 Ground Truth Pfad:", DEFAULT_GT)
pred_dir = st.text_input("📁 Prediction Pfad:", DEFAULT_PRED)

if st.button("🚀 Auswertung starten"):
    if not os.path.exists(gt_dir):
        st.error(f"❌ Ground Truth Pfad nicht gefunden: {gt_dir}")
    elif not os.path.exists(pred_dir):
        st.error(f"❌ Prediction Pfad nicht gefunden: {pred_dir}")
    else:
        df = evaluate_model(gt_dir, pred_dir)
        if df.empty:
            st.warning("⚠️ Keine Labels gefunden! Bitte Pfade prüfen.")
        else:
            st.success("✅ Auswertung abgeschlossen!")

            # -----------------------
            # Rohdaten
            # -----------------------
            st.subheader("📋 Rohdaten")
            st.markdown("""
Jede Zeile steht für **eine Schraube**, die die KI erkannt oder die Ground Truth enthält:
- **image:** Bilddatei  
- **iou:** Überlappung zwischen Vorhersage und Ground Truth (0–1)  
- **gt_class:** Ground Truth Klasse (0=vorhanden, 1=fehlend)  
- **pred_class:** KI-Vorhersage  
- **match:** True=korrekt erkannt
""")
            st.dataframe(df.head(20))

            # -----------------------
            # Gesamtkennzahlen
            # -----------------------
            st.subheader("📊 Gesamtkennzahlen")
            correct = df[df["match"]]
            precision = len(correct) / len(df[df["pred_class"].notna()]) if len(df[df["pred_class"].notna()]) else 0
            recall = len(correct) / len(df[df["gt_class"].notna()]) if len(df[df["gt_class"].notna()]) else 0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            mean_iou = df["iou"].mean()

            st.metric("Precision (Treffsicherheit)", f"{precision:.2f}", help="Wie viele von den erkannten Boxen waren korrekt?")
            st.metric("Recall (Vollständigkeit)", f"{recall:.2f}", help="Wie viele der vorhandenen Boxen wurden korrekt erkannt?")
            st.metric("F1-Score", f"{f1:.2f}", help="Kombination aus Precision und Recall")
            st.metric("Mittlere IoU", f"{mean_iou:.2f}", help="Wie genau die Boxen übereinstimmen")

            # -----------------------
            # Klassen-spezifische Kennzahlen
            # -----------------------
            st.subheader("🔹 Klassen-spezifische Kennzahlen")

            # Bolt (0)
            gt_bolt = df[df["gt_class"] == 0]
            correct_bolt = gt_bolt[gt_bolt["match"]]
            recall_bolt = len(correct_bolt) / len(gt_bolt) if len(gt_bolt) > 0 else 0
            st.metric("Recall vorhandene Schrauben (Bolt)", f"{recall_bolt:.2f}", help="Wie viele vorhandene Schrauben wurden korrekt erkannt?")

            # Missing Bolt (1)
            gt_missing = df[df["gt_class"] == 1]
            correct_missing = gt_missing[gt_missing["match"]]
            recall_missing = len(correct_missing) / len(gt_missing) if len(gt_missing) > 0 else 0
            st.metric("Recall fehlende Schrauben (Missing Bolt)", f"{recall_missing:.2f}", help="Wie viele fehlende Schrauben wurden korrekt erkannt?")

            # -----------------------
            # Konfusionsmatrix
            # -----------------------
            st.subheader("🧠 Konfusionsmatrix")
            cm = pd.crosstab(df["gt_class"], df["pred_class"], rownames=['Tatsächlich'], colnames=['Vorhergesagt'])
            st.dataframe(cm)
            st.markdown("""
Interpretation:
- Oben links: Bolt korrekt erkannt  
- Oben rechts: Bolt fälschlicherweise als Missing erkannt  
- Unten links: Missing Bolt fälschlicherweise als Bolt erkannt  
- Unten rechts: Missing Bolt korrekt erkannt
""")

            # -----------------------
            # IoU-Verteilung
            # -----------------------
            st.subheader("📈 Verteilung der IoU")
            fig, ax = plt.subplots()
            sns.histplot(df["iou"], bins=20, kde=True, ax=ax)
            ax.set_xlabel("IoU (Überlappung)")
            ax.set_ylabel("Anzahl")
            ax.set_title("Genauigkeit der Boxenpositionen")
            st.pyplot(fig)
            st.markdown("""
- Werte nahe 1: Boxen exakt getroffen  
- Werte unter 0.5: KI Box ist deutlich verschoben  
- Sehr niedrige Werte: KI hat Box stark verfehlt oder Boxgröße passt nicht
""")

            # -----------------------
            # Gesamtfazit
            # -----------------------
            st.divider()
            st.markdown("""
### 🧩 Fazit:
Dieses Dashboard zeigt alle relevanten Kennzahlen, sowohl **gesamt** als auch **klassen-spezifisch**.  

- Precision → Treffsicherheit der KI  
- Recall → Vollständigkeit der Erkennung  
- Klassen-Recall → speziell für Bolt und Missing Bolt getrennt  
- Konfusionsmatrix → zeigt Fehlklassifikationen  
- IoU-Verteilung → Qualität der Box-Positionen

So kannst du genau analysieren, **wie zuverlässig die KI fehlende Schrauben erkennt** und wo Verbesserungen nötig sind.
""")
