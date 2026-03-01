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
    Jede TXT-Datei enthält Zeilen im Format: [class x_center y_center width height (conf)]
    Rückgabe: dict mit Bildname als key und np.array mit Boxen als value
    """
    labels = {}
    for path in glob(os.path.join(label_dir, "*.txt")):
        name = os.path.basename(path)
        with open(path, "r") as f:
            data = []
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                # Akzeptiere 5 (ohne Conf) oder 6 (mit Conf) Werte
                if len(parts) >= 5:
                    # Wir speichern nur die ersten 5 Werte [class, x, y, w, h] für die Auswertung
                    data.append(parts[:5])
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


def calc_center_distance(box1, box2):
    """Berechnet die Distanz zwischen den Mittelpunkten zweier Boxen (für Cluster-Logik)."""
    # box: [class, x, y, w, h] -> wir nutzen x(1), y(2)
    return np.linalg.norm(np.array(box1[1:3]) - np.array(box2[1:3]))


def evaluate_model(gt_dir, pred_dir, iou_threshold=0.5, dist_threshold=0.025):
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
        
        # Tracking Arrays für Matches
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        pred_matched = np.zeros(len(pred_boxes), dtype=bool)
        
        # Helper Funktion für Matching
        def find_best_match_for_gt(gt_idx, same_class_only=False):
            gt_box = gt_boxes[gt_idx]
            gt_class = int(gt_box[0])
            
            best_iou_val = -1
            best_dist_val = float('inf')
            best_pred_idx = -1
            
            for i, pred_box in enumerate(pred_boxes):
                if pred_matched[i]: continue # Bereits vergeben
                
                pred_class = int(pred_box[0])
                if same_class_only and pred_class != gt_class: continue
                
                iou = bbox_iou(gt_box[1:], pred_box[1:])
                dist = calc_center_distance(gt_box, pred_box)
                
                # Priorität 1: IoU Match
                if iou >= iou_threshold:
                    if iou > best_iou_val:
                        best_iou_val = iou
                        best_pred_idx = i
                # Priorität 2: Distanz Match (Fallback)
                elif best_iou_val == -1 and dist <= dist_threshold:
                    if dist < best_dist_val:
                        best_dist_val = dist
                        best_pred_idx = i
            return best_pred_idx

        # PASS 1: Match SAME CLASS (Priorität wie in evaluate.py)
        for i in range(len(gt_boxes)):
            best_pred_idx = find_best_match_for_gt(i, same_class_only=True)
            if best_pred_idx != -1:
                gt_matched[i] = True
                pred_matched[best_pred_idx] = True
                stats.append({
                    "image": img_name,
                    "iou": bbox_iou(gt_boxes[i][1:], pred_boxes[best_pred_idx][1:]),
                    "gt_class": int(gt_boxes[i][0]),
                    "pred_class": int(pred_boxes[best_pred_idx][0]),
                    "match": True
                })

        # PASS 2: Match ANY CLASS (Falsch klassifiziert)
        for i in range(len(gt_boxes)):
            if not gt_matched[i]:
                best_pred_idx = find_best_match_for_gt(i, same_class_only=False)
                if best_pred_idx != -1:
                    gt_matched[i] = True
                    pred_matched[best_pred_idx] = True
                    stats.append({
                        "image": img_name,
                        "iou": bbox_iou(gt_boxes[i][1:], pred_boxes[best_pred_idx][1:]),
                        "gt_class": int(gt_boxes[i][0]),
                        "pred_class": int(pred_boxes[best_pred_idx][0]),
                        "match": False # Wrong Class
                    })
                else:
                    # Missed (FN)
                    stats.append({
                        "image": img_name,
                        "iou": 0,
                        "gt_class": int(gt_boxes[i][0]),
                        "pred_class": None,
                        "match": False
                    })

        # Vorhersagen ohne zugeordnetes GT (FP)
        for j, pred_box in enumerate(pred_boxes):
            if not pred_matched[j]:
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
# ABSOLUTE PFADE (FIXED LOCATIONS)
# ============================================================

EVAL_BASE = r"C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\YOLO\testing\infer_train_30_epochs_m_conf(0.4)"

# Ground Truth (immer gleich)
DEFAULT_GT = r"C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\labels\val"


# ============================================================
# DASHBOARD
# ============================================================

# ============================================================
# DASHBOARD
# ============================================================

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
        pred_dir = os.path.join(EVAL_BASE, eval_name)

        st.subheader(f"📁 Evaluierung: `{eval_name}`")
        st.markdown(f"**Pfad:** `{pred_dir}`")

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
            # METRIKEN (globale Berechnung)
            # ============================================================
            tp_total = 0
            wrong_total = 0
            missed_total = 0
            fp_total = 0

            for class_id in sorted(df["gt_class"].dropna().unique()):

                gt_class_df = df[df["gt_class"] == class_id]
                pred_class_df = df[df["pred_class"] == class_id]

                # TP
                tp_c = gt_class_df[gt_class_df["match"]]

                # Falsch klassifiziert
                wrong_c = gt_class_df[
                    (gt_class_df["pred_class"].notna()) &
                    (~gt_class_df["match"])
                ]

                # Nicht erkannt
                missed_c = gt_class_df[
                    gt_class_df["pred_class"].isna()
                ]

                # Überflüssige Boxen
                fp_c = pred_class_df[
                    (pred_class_df["gt_class"].isna()) |
                    (~pred_class_df["match"])
                ]

                tp_total += len(tp_c)
                wrong_total += len(wrong_c)
                missed_total += len(missed_c)
                fp_total += len(fp_c)

            # --- Berechnungen intern immer genau ---
            # Precision = TP / (TP + FP). Wrong Class (GT perspective) gehört NICHT in den Nenner der Precision.
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
            recall = tp_total / (tp_total + wrong_total + missed_total) if (tp_total + wrong_total + missed_total) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            mean_iou = df["iou"].mean()

            # --- Ausgabe gerundet auf 2 Nachkommastellen ---
            with st.expander("📘 Gesamtmetriken & Erklärungen", expanded=True):
                st.markdown("""
                ### 📈 **Gesamtmetriken – Verständlich erklärt**

                - 🎯 **Precision**: Anteil korrekt erkannter Objekte an allen Boxen  
                - ✅ **Recall**: Anteil korrekt erkannter Ground Truths  
                - ⚖️ **F1-Score**: Harmonisches Mittel von Precision & Recall  
                - 📏 **Mittlere IoU**: Durchschnittliche Überlappung
                """)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 Precision*", f"{precision*100:.2f} %")
                col2.metric("✅ Recall*", f"{recall*100:.2f} %")
                col3.metric("⚖️ F1-Score", f"{f1*100:.2f} %")
                col4.metric("📏 Mittlere IoU", f"{mean_iou*100:.2f} %")

            # ============================================================
            # Klassenweise Analyse
            # ============================================================
            st.subheader("🔹 Klassenweise Analyse")

            for class_id, class_name in [(0, "Bolt (vorhanden)"), (1, "Missing Bolt (fehlend)")]:
                
                gt_class_df = df[df["gt_class"] == class_id]
                pred_class_df = df[df["pred_class"] == class_id]

                tp_c = gt_class_df[gt_class_df["match"]]
                wrong_c = gt_class_df[(gt_class_df["pred_class"].notna()) & (~gt_class_df["match"])]
                missed_c = gt_class_df[gt_class_df["pred_class"].isna()]
                false_positive_c = pred_class_df[(pred_class_df["gt_class"].isna()) | (~pred_class_df["match"])]

                tp_val = len(tp_c)
                wrong_val = len(wrong_c)
                missed_val = len(missed_c)
                fp_val = len(false_positive_c)

                # --- Berechnungen intern genau ---
                precision_c = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
                recall_c = tp_val / (tp_val + wrong_val + missed_val) if (tp_val + wrong_val + missed_val) > 0 else 0
                f1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c + 1e-8)
                missed_percent = (missed_val / len(gt_class_df) * 100) if len(gt_class_df) > 0 else 0

                # --- Anzeige auf 2 Nachkommastellen ---
                with st.expander(f"📊 Detaillierte Analyse für {class_name}", expanded=False):
                    st.markdown(f"""
                    **🔩 {class_name}**

                    - ✅ Korrekt erkannt: {tp_val}  
                    - ⚠️ Falsch klassifiziert: {wrong_val}  
                    - ❌ Nicht erkannt: {missed_val} ({missed_percent:.2f}% der Ground Truths)  
                    - 📦 Überflüssige Boxen: {fp_val}
                    """)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🎯 Precision*", f"{precision_c*100:.2f} %")
                    col2.metric("✅ Recall*", f"{recall_c*100:.2f} %")
                    col3.metric("⚖️ F1-Score", f"{f1_c*100:.2f} %")
                    col4.metric("❌ Nicht erkannt", f"{missed_percent:.2f} %")



            # Konfusionsmatrix
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

            # IoU-Verteilung
            with st.expander("📈 IoU-Verteilung", expanded=False):
                fig, ax = plt.subplots()
                sns.histplot(df["iou"], bins=20, kde=True, ax=ax)
                ax.set_xlabel("IoU (0–1)")
                ax.set_ylabel("Anzahl Boxen")
                ax.set_title("Verteilung der IoU-Werte")
                st.pyplot(fig)
