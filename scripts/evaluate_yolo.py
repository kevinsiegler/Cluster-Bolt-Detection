import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------

def load_labels(label_dir):
    """Lädt alle YOLO-Label-Dateien aus einem Ordner"""
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
    """Berechnet IoU (Intersection over Union) zwischen zwei Bounding Boxes"""
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
    """Vergleicht Ground Truth und Prediction Dateien"""
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

        # nicht gematchte Predictions = False Positives
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


# ---------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------

if __name__ == "__main__":
    gt_dir = "../dataset/labels/val"
    pred_dir = "runs/detect/predict/labels"


    print("🔍 Starte Auswertung ...")
    df = evaluate_model(gt_dir, pred_dir)

    # Grundstatistiken
    correct = df[df["match"]]
    precision = len(correct) / len(df[df["pred_class"].notna()]) if len(df[df["pred_class"].notna()]) else 0
    recall = len(correct) / len(df[df["gt_class"].notna()]) if len(df[df["gt_class"].notna()]) else 0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    mean_iou = df["iou"].mean()

    print("\n📊  --- AUSWERTUNG ---")
    print(f"Anzahl aller Ground Truth Boxen: {len(df[df['gt_class'].notna()])}")
    print(f"Anzahl aller Predictions:        {len(df[df['pred_class'].notna()])}")
    print(f"Treffer (korrekt erkannt):       {len(correct)}")
    print(f"Precision:                       {precision:.3f}")
    print(f"Recall:                          {recall:.3f}")
    print(f"F1-Score:                        {f1:.3f}")
    print(f"Durchschnittliche IoU:           {mean_iou:.3f}")

    # Confusion-Matrix
    cm = pd.crosstab(df["gt_class"], df["pred_class"], rownames=['GT'], colnames=['Prediction'])
    print("\nKonfusionsmatrix:")
    print(cm)

    # Ergebnisse für jede Klasse
    print("\nErgebnisse pro Klasse:")
    for c in [0, 1]:
        subset = df[df["gt_class"] == c]
        tp = len(subset[subset["match"]])
        fn = len(subset[~subset["match"] & subset["pred_class"].isna()])
        fp = len(df[(df["gt_class"].isna()) & (df["pred_class"] == c)])
        print(f"Klasse {c} → TP: {tp}, FN: {fn}, FP: {fp}")

    # Diagramme
    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.histplot(df["iou"], bins=20, kde=True)
    plt.title("IoU-Verteilung")
    plt.xlabel("IoU")
    plt.ylabel("Häufigkeit")
    plt.show()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Konfusionsmatrix")
    plt.show()

    print("\n✅ Auswertung abgeschlossen.")
