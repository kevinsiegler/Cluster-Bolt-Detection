import os
import numpy as np
import pandas as pd
from glob import glob
import plotly.graph_objects as go
import streamlit as st
import re

# ============================================================
# LOGIK (EXAKT IDENTISCH)
# ============================================================

def load_labels(label_dir):
    labels = {}
    for path in glob(os.path.join(label_dir, "*.txt")):
        name = os.path.basename(path)
        data = []
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        data.append(parts[:5])
        labels[name] = np.array(data) if len(data) > 0 else np.empty((0, 5))
    return labels

def bbox_iou(box1, box2):
    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    inter_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * \
                 max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_model_logic(gt_labels, pred_labels, iou_threshold=0.5):
    stats = []
    all_imgs = set(gt_labels.keys()).union(set(pred_labels.keys()))
    for img_name in all_imgs:
        gt_boxes = gt_labels.get(img_name, np.empty((0, 5)))
        pred_boxes = pred_labels.get(img_name, np.empty((0, 5)))
        matched_pred = set()

        if gt_boxes.size > 0:
            for gt_box in gt_boxes:
                gt_class, best_iou, best_pred_idx = int(gt_box[0]), 0, None
                for i, pred_box in enumerate(pred_boxes):
                    iou = bbox_iou(gt_box[1:], pred_box[1:])
                    if iou > best_iou:
                        best_iou, best_pred_idx = iou, i
                if best_iou >= iou_threshold and best_pred_idx is not None:
                    matched_pred.add(best_pred_idx)
                    stats.append({
                        "gt_class": gt_class,
                        "pred_class": int(pred_boxes[best_pred_idx][0]),
                        "match": gt_class == int(pred_boxes[best_pred_idx][0])
                    })
                else:
                    stats.append({"gt_class": gt_class, "pred_class": None, "match": False})

        for j, pred_box in enumerate(pred_boxes):
            if j not in matched_pred:
                stats.append({"gt_class": None, "pred_class": int(pred_box[0]), "match": False})

    return pd.DataFrame(stats)

def calculate_metrics(df, class_id=None):
    d = df[df["gt_class"] == class_id] if class_id is not None else df[df["gt_class"].notna()]
    tp = len(d[d["match"]])
    wrong = len(d[(d["pred_class"].notna()) & (~d["match"])])
    missed = len(d[d["pred_class"].isna()])

    p_df = df[df["pred_class"] == class_id] if class_id is not None else df[df["pred_class"].notna()]
    fp = len(p_df[(p_df["gt_class"].isna()) | (~p_df["match"])])

    rec = tp / (tp + wrong + missed) if (tp + wrong + missed) > 0 else 0
    prec = tp / (tp + fp + wrong + missed) if (tp + fp + wrong + missed) > 0 else 0
    return prec, rec

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="YOLOv8 Trainingsdaten-Analyse", layout="wide")
st.title("📊 Einfluss der Trainingsdatenmenge auf die Modellgenauigkeit")

# Sidebar Gewichtung
st.sidebar.header("⚖️ Gewichtung (F-Beta)")
weight_mode = st.sidebar.select_slider(
    "Fokus",
    options=["Precision (Keine Fehler)", "Ausgewogen (F1)", "Recall (Alles finden)"],
    value="Ausgewogen (F1)"
)

beta = {"Precision (Keine Fehler)": 0.5, "Ausgewogen (F1)": 1.0, "Recall (Alles finden)": 2.0}[weight_mode  ]
st.sidebar.info(f"Beta = {beta}")

# ============================================================
# PFADSETUP
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_BASE = os.path.join(BASE_DIR, "scripts", "runs", "detect", "evaluations_w_different_amounts_data_conf(0.2)")
GT_DIR = os.path.join(BASE_DIR, "dataset", "labels", "val")

gt_labels = load_labels(GT_DIR)
eval_folders = [d for d in os.listdir(EVAL_BASE) if os.path.isdir(os.path.join(EVAL_BASE, d))]

results = []

# ============================================================
# AUSWERTUNG ALLER MODELLE
# ============================================================

for folder in eval_folders:
    # ➜ Prozent aus Ordnernamen extrahieren
    m = re.search(r"subset_(\d+)_n", folder)
    if not m:
        continue

    percent = int(m.group(1))
    train_frac = percent / 100.0

    df_ev = evaluate_model_logic(
        gt_labels,
        load_labels(os.path.join(EVAL_BASE, folder, "labels"))
    )

    p_t, r_t = calculate_metrics(df_ev)
    p_b, r_b = calculate_metrics(df_ev, 0)
    p_m, r_m = calculate_metrics(df_ev, 1)

    results.append({
        "train_frac": train_frac,
        "Total_P": p_t, "Total_R": r_t,
        "Bolt_P": p_b, "Bolt_R": r_b,
        "Missing_P": p_m, "Missing_R": r_m
    })

df = pd.DataFrame(results).sort_values("train_frac")

# ============================================================
# F-BETA
# ============================================================

def f_beta(p, r, b):
    return (1 + b**2) * (p * r) / ((b**2 * p) + r + 1e-8)

df["Total_F"] = f_beta(df["Total_P"], df["Total_R"], beta)
df["Bolt_F"] = f_beta(df["Bolt_P"], df["Bolt_R"], beta)
df["Missing_F"] = f_beta(df["Missing_P"], df["Missing_R"], beta)

# ============================================================
# PLOT
# ============================================================

def create_plot(df, p_col, r_col, f_col, title):
    best = df.loc[df[f_col].idxmax()]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["train_frac"], y=df[p_col], name="Precision"))
    fig.add_trace(go.Scatter(x=df["train_frac"], y=df[r_col], name="Recall"))
    fig.add_trace(go.Scatter(x=df["train_frac"], y=df[f_col], name="F-Beta", line=dict(dash="dot", width=4)))

    fig.add_vline(x=best["train_frac"], line_dash="dash", line_color="gold")
    fig.add_trace(go.Scatter(
        x=[best["train_frac"]],
        y=[best[f_col]],
        mode="markers+text",
        marker=dict(symbol="star", size=14, color="gold"),
        text=[f"Optimum: {int(best['train_frac']*100)}%"],
        textposition="top center"
    ))

    fig.update_layout(
        title=title,
        hovermode="x unified",
        xaxis_title="Anteil der Trainingsdaten (0–1)",
        yaxis=dict(range=[-0.05, 1.05]),
        template="plotly_white"
    )
    return fig

t1, t2, t3 = st.tabs(["Gesamt", "Bolt", "Missing Bolt"])
with t1: st.plotly_chart(create_plot(df, "Total_P", "Total_R", "Total_F", "Gesamtmetrik"), use_container_width=True)
with t2: st.plotly_chart(create_plot(df, "Bolt_P", "Bolt_R", "Bolt_F", "Klasse: Bolt"), use_container_width=True)
with t3: st.plotly_chart(create_plot(df, "Missing_P", "Missing_R", "Missing_F", "Klasse: Missing Bolt"), use_container_width=True)

st.dataframe(df.style.highlight_max(subset=["Total_F", "Bolt_F", "Missing_F"], color="gold"))
