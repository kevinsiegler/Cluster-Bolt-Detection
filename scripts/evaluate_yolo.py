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
EVAL_BASE = os.path.join(BASE_DIR, "scripts", "runs", "detect", "evaluations")

# Ground Truth bleibt gleich
DEFAULT_GT = os.path.join(BASE_DIR, "dataset", "labels", "val")

# Streamlit Layout
st.set_page_config(page_title="YOLOv8 Schrauben-Dashboard", layout="wide")
st.title("🔩 YOLOv8 Dashboard – Schraubenanalyse (Multi-Evaluation)")

st.write("""
Dieses Dashboard zeigt mehrere KI-Auswertungen nebeneinander.  
Wähle oben die gewünschte **Evaluierung** aus, um deren Analyse zu sehen.
""")

# Alle Evaluierungsordner automatisch auflisten
if not os.path.exists(EVAL_BASE):
    st.warning(f"Kein Evaluations-Ordner gefunden unter: `{EVAL_BASE}`")
    st.stop()

available_evals = sorted([d for d in os.listdir(EVAL_BASE)
                          if os.path.isdir(os.path.join(EVAL_BASE, d))])

if not available_evals:
    st.warning("⚠️ Keine vorhandenen Evaluierungen gefunden!")
    st.stop()

# Tabs für jede Evaluierung
tabs = st.tabs(available_evals)

for i, eval_name in enumerate(available_evals):
    with tabs[i]:
        pred_dir = os.path.join(EVAL_BASE, eval_name, "labels")

        st.subheader(f"📁 Evaluierung: `{eval_name}`")
        st.markdown(f"**Pfad:** `{pred_dir}`")

        # === dein bisheriger Code ab hier bleibt exakt gleich ===
        if not os.path.exists(DEFAULT_GT):
            st.error(f"❌ Ground Truth Pfad nicht gefunden: {DEFAULT_GT}")
        elif not os.path.exists(pred_dir):
            st.error(f"❌ Prediction Pfad nicht gefunden: {pred_dir}")
        else:
            df = evaluate_model(DEFAULT_GT, pred_dir)
            if df.empty:
                st.warning("⚠️ Keine Labels gefunden!")
            else:
                st.success("✅ Auswertung abgeschlossen!")

            # ============================================================
            # METRIKEN
            # ============================================================
            gt_nonempty = df[df["gt_class"].notna()]
            pred_nonempty = df[df["pred_class"].notna()]

            tp_correct = df[(df["gt_class"].notna()) & (df["pred_class"].notna()) & (df["match"])]
            wrong_class = df[(df["gt_class"].notna()) & (df["pred_class"].notna()) & (~df["match"])]
            missed = df[(df["gt_class"].notna()) & (df["pred_class"].isna())]
            false_positives = df[(df["gt_class"].isna()) & (df["pred_class"].notna())]

            # Erweiterte Precision und Recall
            recall = len(tp_correct) / (len(tp_correct) + len(wrong_class) + len(missed)) if len(gt_nonempty) > 0 else 0
            precision = len(tp_correct) / (
                len(tp_correct) + len(wrong_class) + len(false_positives) + len(missed)
            ) if (len(gt_nonempty) + len(pred_nonempty)) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            mean_iou = df["iou"].mean()

            # ------------------------------------------------------------
            # Gesamtmetriken mit Erklärungen
            # ------------------------------------------------------------
            with st.expander("📘 Gesamtmetriken & Erklärungen", expanded=True):
                st.markdown("""
                ### 📈 **Gesamtmetriken – Verständlich erklärt**

                - 🎯 **Erweiterte Precision (Genauigkeit)**  
                  Anteil korrekt erkannter Objekte an **allen Boxen (inkl. verfehlter & überflüssiger)**.  
                  Zeigt, wie „sauber“ das Gesamtergebnis ist.

                - ✅ **Erweiterter Recall (Vollständigkeit)**  
                  Anteil korrekt erkannter Ground Truths.  
                  Zeigt, wie viele echte Objekte die KI wirklich gefunden hat.

                - ⚖️ **F1-Score:**  
                  Harmonisches Mittel von Precision und Recall – kombiniert Zuverlässigkeit & Vollständigkeit.

                - 📏 **Mittlere IoU:**  
                  Durchschnittliche Überlappung (Intersection over Union) zwischen Ground Truth & Prediction.
                """)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 Precision*", f"{precision*100:.2f} %")
                col2.metric("✅ Recall*", f"{recall*100:.2f} %")
                col3.metric("⚖️ F1-Score", f"{f1*100:.2f} %")
                col4.metric("📏 Mittlere IoU", f"{mean_iou*100:.2f} %")

            # ------------------------------------------------------------
            # Klassenweise Analyse
            # ------------------------------------------------------------
            st.subheader("🔹 Klassenweise Analyse")

            for class_id, class_name in [(0, "Bolt (vorhanden)"), (1, "Missing Bolt (fehlend)")]:
                gt_class_df = df[df["gt_class"] == class_id]
                tp_c = gt_class_df[gt_class_df["match"]]
                wrong_c = gt_class_df[(gt_class_df["pred_class"].notna()) & (~gt_class_df["match"])]
                missed_c = gt_class_df[gt_class_df["pred_class"].isna()]
                pred_class_df = df[df["pred_class"] == class_id]
                fp_c = pred_class_df[(pred_class_df["gt_class"].isna()) | (~pred_class_df["match"])]

                recall_c = len(tp_c) / (len(tp_c) + len(wrong_c) + len(missed_c)) if len(gt_class_df) > 0 else 0
                precision_c = len(tp_c) / (
                    len(tp_c) + len(fp_c) + len(wrong_c) + len(missed_c)
                ) if (len(gt_class_df) + len(pred_class_df)) > 0 else 0
                f1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c + 1e-8)

                missed_count = len(missed_c)
                missed_percent = (missed_count / len(gt_class_df) * 100) if len(gt_class_df) > 0 else 0

                with st.expander(f"📊 Detaillierte Analyse für {class_name}", expanded=False):
                    st.markdown(f"""
                    **🔩 {class_name}**

                    - ✅ **Korrekt erkannt:** {len(tp_c)}  
                    - ⚠️ **Falsch klassifiziert:** {len(wrong_c)}  
                    - ❌ **Nicht erkannt:** {missed_count} ({missed_percent:.2f}% der Ground Truths)  
                    - 📦 **Überflüssige Boxen:** {len(fp_c)}

                    **Erklärungen:**
                    - 🎯 **Precision*** = Wie genau alle Boxen (inkl. Fehler & fehlender) insgesamt waren.  
                    - ✅ **Recall*** = Wie viele Ground Truths richtig erkannt wurden.  
                    - ⚖️ **F1-Score** = Kombination aus beiden für ausgewogene Bewertung.
                    """)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🎯 Precision*", f"{precision_c*100:.2f} %")
                    col2.metric("✅ Recall*", f"{recall_c*100:.2f} %")
                    col3.metric("⚖️ F1-Score", f"{f1_c*100:.2f} %")
                    col4.metric("❌ Nicht erkannt", f"{missed_percent:.2f} %")

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # Konfusionsmatrix
            # ------------------------------------------------------------
            with st.expander("🧠 Konfusionsmatrix", expanded=False):
                cm = pd.crosstab(df["gt_class"], df["pred_class"],
                                rownames=['Tatsächlich (Ground Truth)'],
                                colnames=['Vorhergesagt (Prediction)']).fillna(0)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
                ax.set_xlabel("Vorhergesagte Klasse")
                ax.set_ylabel("Tatsächliche Klasse")
                ax.set_title("Konfusionsmatrix der Klassenzuordnung")
                st.pyplot(fig)

                st.markdown("""
                Die **Konfusionsmatrix** zeigt, wie gut die KI die beiden Klassen unterscheiden kann:

                | Zelle | Bedeutung |
                |-------|------------|
                | **[0,0]** | ✅ Bolt korrekt erkannt |
                | **[1,1]** | ✅ Missing Bolt korrekt erkannt |
                | **[0,1]** | ⚠️ Bolt fälschlich als Missing Bolt erkannt |
                | **[1,0]** | ⚠️ Missing Bolt fälschlich als Bolt erkannt |

                🔹 **Interpretation:**  
                - Hohe Werte auf der Diagonalen → gute Klassentrennung.  
                - Werte außerhalb der Diagonalen → Verwechslungen zwischen den Klassen.
                """)

            # ------------------------------------------------------------
            # IoU-Verteilung
            # ------------------------------------------------------------
            with st.expander("📈 IoU-Verteilung", expanded=False):
                fig, ax = plt.subplots()
                sns.histplot(df["iou"], bins=20, kde=True, ax=ax)
                ax.set_xlabel("IoU (0–1)")
                ax.set_ylabel("Anzahl Boxen")
                ax.set_title("Verteilung der IoU-Werte")
                st.pyplot(fig)

                st.markdown("""
                💡 **Interpretation:**  
                - Hohe IoU-Werte (>0.75) = sehr genaue Positionierung der Box.  
                - Niedrige IoU-Werte (<0.3) = Box sitzt zu weit entfernt oder hat falsche Größe.
                """)
