import os
import numpy as np
import pandas as pd
from glob import glob
import plotly.graph_objects as go
import streamlit as st
import re

# ============================================================
# LOGIK (Bleibt identisch zu deiner stabilen Basis)
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
                    if len(parts) >= 5: data.append(parts[:5])
        labels[name] = np.array(data) if len(data) > 0 else np.empty((0, 5))
    return labels

def bbox_iou(box1, box2):
    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    inter_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_model_logic(gt_labels, pred_labels, iou_threshold=0.5):
    stats = []
    all_imgs = set(gt_labels.keys()).union(set(pred_labels.keys()))
    for img_name in all_imgs:
        gt_boxes, pred_boxes = gt_labels.get(img_name, np.empty((0, 5))), pred_labels.get(img_name, np.empty((0, 5)))
        matched_pred = set()
        if gt_boxes.size > 0:
            for gt_box in gt_boxes:
                gt_class, best_iou, best_pred_idx = int(gt_box[0]), 0, None
                for i, pred_box in enumerate(pred_boxes):
                    iou = bbox_iou(gt_box[1:], pred_box[1:])
                    if iou > best_iou: best_iou, best_pred_idx = iou, i
                if best_iou >= iou_threshold and best_pred_idx is not None:
                    matched_pred.add(best_pred_idx)
                    stats.append({"gt_class": gt_class, "pred_class": int(pred_boxes[best_pred_idx][0]), "match": gt_class == int(pred_boxes[best_pred_idx][0])})
                else: stats.append({"gt_class": gt_class, "pred_class": None, "match": False})
        for j, pred_box in enumerate(pred_boxes):
            if j not in matched_pred: stats.append({"gt_class": None, "pred_class": int(pred_box[0]), "match": False})
    return pd.DataFrame(stats)

def calculate_metrics(df, class_id=None):
    d = df[df["gt_class"] == class_id] if class_id is not None else df[df["gt_class"].notna()]
    tp = len(d[d["match"]])
    wrong = len(d[(d["pred_class"].notna()) & (~d["match"])])
    missed = len(d[d["pred_class"].isna()])
    p_df = df[df["pred_class"] == class_id] if class_id is not None else df[df["pred_class"].notna()]
    fp = len(p_df[(p_df["gt_class"].isna()) | (~p_df["match"])])
    
    # Original Metrik-Logik
    rec = tp / (tp + wrong + missed) if (tp + wrong + missed) > 0 else 0
    prec = tp / (tp + fp + wrong + missed) if (tp + fp + wrong + missed) > 0 else 0
    return prec, rec

# ============================================================
# STREAMLIT UI & INTERACTION
# ============================================

st.set_page_config(page_title="YOLOv8 Optimal Confidence", layout="wide")
st.title("🔩 Analyse der optimalen Confidence")

# Sidebar für Gewichtung
st.sidebar.header("⚖️ Gewichtung (F-Beta Score)")
weight_mode = st.sidebar.select_slider(
    "Was ist wichtiger?",
    options=["Precision (Keine Fehler)", "Ausgewogen (F1)", "Recall (Alles finden)"],
    value="Ausgewogen (F1)"
)

# Mapping des Sliders auf den Beta-Wert
beta = 1.0
if weight_mode == "Precision (Keine Fehler)": beta = 0.5
elif weight_mode == "Recall (Alles finden)": beta = 2.0

st.sidebar.info(f"Aktueller Fokus: **{weight_mode}** (Beta={beta})")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_BASE = os.path.join(BASE_DIR, "scripts", "runs", "detect", "evaluations")
DEFAULT_GT = os.path.join(BASE_DIR, "dataset", "labels", "val")

eval_folders = [d for d in os.listdir(EVAL_BASE) if os.path.isdir(os.path.join(EVAL_BASE, d))]
gt_labels = load_labels(DEFAULT_GT)
results = []

for folder in eval_folders:
    conf_m = re.findall(r"0\.\d+|1\.0", folder)
    if not conf_m: continue
    c_val = float(conf_m[0])
    df_ev = evaluate_model_logic(gt_labels, load_labels(os.path.join(EVAL_BASE, folder, "labels")))
    
    p_t, r_t = calculate_metrics(df_ev)
    p_b, r_b = calculate_metrics(df_ev, 0)
    p_m, r_m = calculate_metrics(df_ev, 1)
    
    results.append({"conf": c_val, "Total_P": p_t, "Total_R": r_t, "Bolt_P": p_b, "Bolt_R": r_b, "Missing_P": p_m, "Missing_R": r_m})

# 0.95 & 1.0 Auffüllen
for c_req in [round(x, 2) for x in np.arange(0.1, 1.05, 0.05)]:
    if not any(abs(res['conf'] - c_req) < 0.01 for res in results):
        results.append({k: (c_req if k=="conf" else 0.0) for k in ["conf", "Total_P", "Total_R", "Bolt_P", "Bolt_R", "Missing_P", "Missing_R"]})

df = pd.DataFrame(results).sort_values("conf")

# F-Beta Berechnung für alle Spalten
def f_beta(p, r, b):
    return (1 + b**2) * (p * r) / ((b**2 * p) + r + 1e-8)

df["Total_F"] = f_beta(df["Total_P"], df["Total_R"], beta)
df["Bolt_F"] = f_beta(df["Bolt_P"], df["Bolt_R"], beta)
df["Missing_F"] = f_beta(df["Missing_P"], df["Missing_R"], beta)

def create_plot(df, p_col, r_col, f_col, title):
    fig = go.Figure()
    # Besten Punkt finden
    best_row = df.loc[df[f_col].idxmax()]
    
    # Linien
    fig.add_trace(go.Scatter(x=df["conf"], y=df[p_col], name="Precision", line=dict(color='#0077b6', width=2)))
    fig.add_trace(go.Scatter(x=df["conf"], y=df[r_col], name="Recall", line=dict(color='#e63946', width=2)))
    fig.add_trace(go.Scatter(x=df["conf"], y=df[f_col], name=f"F-Beta (Score)", line=dict(color='#fb8500', width=4, dash='dot')))
    
    # Goldene Markierung für das Maximum
    fig.add_vline(x=best_row["conf"], line_width=2, line_dash="dash", line_color="#FFD700")
    fig.add_trace(go.Scatter(
        x=[best_row["conf"]], y=[best_row[f_col]],
        mode='markers+text', name='Bester Score',
        marker=dict(color='#FFD700', size=12, symbol='star'),
        text=[f"Optimum: {best_row['conf']}"], textposition="top center"
    ))

    fig.update_layout(
        title=f"{title} <br><sup>Beste Confidence bei {best_row['conf']} (Score: {best_row[f_col]:.3f})</sup>",
        hovermode="x unified", template="plotly_white", yaxis=dict(range=[-0.05, 1.05])
    )
    return fig

t1, t2, t3 = st.tabs(["Gesamtheit", "Bolt", "Missing Bolt"])
with t1: st.plotly_chart(create_plot(df, "Total_P", "Total_R", "Total_F", "Gesamtanalyse"), use_container_width=True)
with t2: st.plotly_chart(create_plot(df, "Bolt_P", "Bolt_R", "Bolt_F", "Klasse: Bolt"), use_container_width=True)
with t3: st.plotly_chart(create_plot(df, "Missing_P", "Missing_R", "Missing_F", "Klasse: Missing Bolt"), use_container_width=True)

st.write("### Berechnete Werte mit Gewichtung")
st.dataframe(df.style.highlight_max(subset=["Total_F", "Bolt_F", "Missing_F"], color="gold"))