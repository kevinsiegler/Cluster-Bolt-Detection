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
    """
    Lädt YOLO-Labels aus einem Verzeichnis.
    Jede TXT-Datei enthält Zeilen im Format: [class x_center y_center width height]
    Rückgabe: dict mit Bildname als key und np.array mit Boxen als value
    """
    labels = {}
    for path in glob(os.path.join(label_dir, "*.txt")):
        name = os.path.basename(path)
        with open(path, "r") as f:
            data = []
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    data.append(parts)
            labels[name] = np.array(data)
    return labels

def bbox_iou(box1, box2):
    """
    Berechnet die IoU (Intersection over Union) zweier Boxen.
    Boxen werden als [x_center, y_center, width, height] übergeben.
    """
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
    """
    Vergleicht Ground Truth Boxen mit Predictions.
    Liefert ein DataFrame mit Details pro Box:
    - image: Bildname
    - iou: IoU Wert
    - gt_class: Ground Truth Klasse
    - pred_class: Vorhergesagte Klasse
    - match: True, wenn korrekt vorhergesagt
    """
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

        # Vorhersagen ohne zugeordnetes GT
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

st.title("🔩 YOLOv8 Dashboard – Schraubenanalyse")
st.write("Bewertet die KI-Leistung bei der Erkennung von **vorhandenen Schrauben (Bolt)** und **fehlenden Schrauben (Missing Bolt)**.")

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
            st.warning("⚠️ Keine Labels gefunden!")
        else:
            st.success("✅ Auswertung abgeschlossen!")

            # -----------------------
            # Ground Truth nach Klassen
            # -----------------------
            gt_bolt = df[df["gt_class"] == 0]
            gt_missing = df[df["gt_class"] == 1]

            # -----------------------
            # Rohdaten
            # -----------------------
            with st.expander("📋 Rohdaten (erste 20 Zeilen)"):
                st.write("Zeigt alle Boxen, sowohl aus Ground Truth als auch aus Predictions.")
                st.dataframe(df.head(20))

            # -----------------------
            # Gesamtkennzahlen
            # -----------------------
            correct = df[df["match"]]
            pred_nonempty = df[df["pred_class"].notna()]
            gt_nonempty = df[df["gt_class"].notna()]

            precision = len(correct) / len(pred_nonempty) if len(pred_nonempty) else 0
            recall = len(correct) / len(gt_nonempty) if len(gt_nonempty) else 0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            mean_iou = df["iou"].mean()

            st.subheader("📊 Gesamtkennzahlen")
            st.markdown("""
**Erklärungen der Metriken:**
- 🎯 **Precision:** Anteil korrekt erkannter Boxen an allen Vorhersagen. Verluste entstehen durch:
  1. Falsch klassifizierte Boxen (vorhanden vs. fehlend)
  2. Vorhersagen ohne Ground Truth
- ✅ **Recall:** Anteil korrekt erkannter Boxen an allen Ground Truth Boxen. Verluste entstehen durch übersehene Boxen.
- 📊 **F1-Score:** Harmonisches Mittel von Precision und Recall.
- 📌 **Mittlere IoU:** Durchschnittliche Überlappung der Boxen (0–1, in % umgerechnet)
""")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Precision", f"{precision*100:.2f} %")
            col2.metric("✅ Recall", f"{recall*100:.2f} %")
            col3.metric("📊 F1-Score", f"{f1*100:.2f} %")
            col4.metric("📌 Mittlere IoU", f"{mean_iou*100:.2f} %")

            # -----------------------
            # Precision-Fehleranalyse
            # -----------------------
            with st.expander("⚠️ Precision-Fehleranalyse"):
                st.write("Falsch-positive Vorhersagen:")
                fp_wrong_class = df[(df["pred_class"].notna()) & (~df["match"]) & (df["gt_class"].notna())]
                fp_no_gt = df[(df["pred_class"].notna()) & (df["gt_class"].isna())]

                st.markdown(f"🟥 Falsch klassifizierte Boxen: {len(fp_wrong_class)} ({len(fp_wrong_class)/len(pred_nonempty)*100:.2f} % der Vorhersagen)")
                st.markdown(f"📦 Vorhersagen ohne Ground Truth: {len(fp_no_gt)} ({len(fp_no_gt)/len(pred_nonempty)*100:.2f} %)")

                bolts_no_gt = fp_no_gt[fp_no_gt["pred_class"] == 0]
                missing_no_gt = fp_no_gt[fp_no_gt["pred_class"] == 1]
                st.markdown(f"🔹 Bolt ohne Ground Truth: {len(bolts_no_gt)}")
                st.markdown(f"🔹 Missing Bolt ohne Ground Truth: {len(missing_no_gt)}")

            # -----------------------
            # Recall-Fehleranalyse
            # -----------------------
            with st.expander("⚠️ Recall-Fehleranalyse"):
                st.write("Falsch-negative Vorhersagen (übersehene Ground Truth Boxen):")
                fn_not_detected = df[(df["gt_class"].notna()) & (~df["match"])]
                fn_bolt = fn_not_detected[fn_not_detected["gt_class"] == 0]
                fn_missing = fn_not_detected[fn_not_detected["gt_class"] == 1]

                st.markdown(f"❌ Gesamt übersehene Boxen: {len(fn_not_detected)} ({len(fn_not_detected)/len(gt_nonempty)*100:.2f} % der Ground Truth)")
                st.markdown(f"🔹 Übersehene Bolt-Boxen: {len(fn_bolt)} ({len(fn_bolt)/len(gt_bolt)*100 if len(gt_bolt)>0 else 0:.2f} %)")
                st.markdown(f"🔹 Übersehene Missing Bolt-Boxen: {len(fn_missing)} ({len(fn_missing)/len(gt_missing)*100 if len(gt_missing)>0 else 0:.2f} %)")

            # -----------------------
            # Klassen-spezifische Recall-Kennzahlen
            # -----------------------
            st.subheader("🔹 Klassen-spezifische Recall-Kennzahlen")
            col1, col2 = st.columns(2)
            recall_bolt = len(gt_bolt[gt_bolt["match"]]) / len(gt_bolt) if len(gt_bolt) > 0 else 0
            recall_missing = len(gt_missing[gt_missing["match"]]) / len(gt_missing) if len(gt_missing) > 0 else 0
            col1.metric("🔩 Bolt Recall", f"{recall_bolt*100:.2f} %", help="Anteil korrekt erkannter vorhandener Schrauben")
            col2.metric("⚠️ Missing Bolt Recall", f"{recall_missing*100:.2f} %", help="Anteil korrekt erkannter fehlender Schrauben")

            # -----------------------
            # Konfusionsmatrix
            # -----------------------
            st.subheader("🧠 Konfusionsmatrix")
            cm = pd.crosstab(df["gt_class"], df["pred_class"], rownames=['Tatsächlich'], colnames=['Vorhergesagt'])
            st.dataframe(cm)

            # -----------------------
            # IoU-Verteilung
            # -----------------------
            st.subheader("📈 IoU-Verteilung")
            fig, ax = plt.subplots()
            sns.histplot(df["iou"], bins=20, kde=True, ax=ax)
            ax.set_xlabel("IoU (0–1)")
            ax.set_ylabel("Anzahl Boxen")
            st.pyplot(fig)
            st.markdown("Höhere Werte → Boxen exakt getroffen, niedrige Werte → starke Abweichung der Boxposition oder Größe.")
