import streamlit as st
import os
import sys
import cv2
import numpy as np
import yaml
import pickle
import random
from scipy.spatial.distance import cdist

# --- Pfad-Management ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) 
from Cluster.utils import load_config, load_yolo_labels

# --- Hilfsfunktionen ---

def find_image_path(image_id, image_folders):
    for folder in image_folders:
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(folder, image_id + ext)
            if os.path.exists(path):
                return path
    return None

def draw_boxes(image, boxes, color, thickness):
    h, w = image.shape[:2]
    if boxes.ndim == 1: boxes = boxes.reshape(1, -1)
    for box in boxes:
        xc, yc, bw, bh = box
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

@st.cache_data(show_spinner="Lade und bewerte SCALED Inferenz-Ergebnisse...", ttl=600)
def load_and_evaluate_all_data():
    """
    Lädt die Ergebnisse eines abgeschlossenen Inferenz-Laufs und evaluiert sie.
    Diese Funktion wird nur einmal ausgeführt und dient der Filterung der Ergebnisse.
    Dies wird nur einmal ausgeführt und dann gecached.
    """
    cfg = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))
    
    # --- Pfade ---
    run_name = cfg['inference'].get('run_name', 'default_run') # Kein _scaled mehr
    pred_dir = os.path.join(cfg['paths']['output_root'], "inference", run_name)
    val_input_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_input")
    val_gt_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_gt")
    inference_input_dir = cfg['paths']['inference_input_dir']
    
    if not os.path.isdir(pred_dir):
        st.error(f"Inferenz-Verzeichnis nicht gefunden: `{pred_dir}`. Bitte führen Sie zuerst `inference_with_scaling.py` aus.")
        return []

    # --- Lade alle relevanten Dateien ---
    eval_files = [f for f in os.listdir(val_gt_dir) if f.endswith('.npy')]
    dist_thresh = cfg['evaluation']['dist_threshold']
    
    all_results = []

    for filename in eval_files:
        image_id = os.path.splitext(filename)[0]
        pred_path = os.path.join(pred_dir, f"{image_id}.txt")
        if not os.path.exists(pred_path): continue # Überspringe, wenn keine Prognose existiert
        
        # --- Lade alle Daten für die Auswertung ---
        # Angleichung an evaluate.py: Robusteres Laden von input_data
        input_data_path = os.path.join(val_input_dir, f"{image_id}.npy")
        input_data = np.empty((0, 4))
        if os.path.exists(input_data_path):
            loaded_data = np.load(input_data_path)
            if loaded_data.ndim == 2 and loaded_data.shape[1] >= 2:
                input_data = loaded_data

        gt_missing_data = np.load(os.path.join(val_gt_dir, filename))
        
        # Lade auch den YOLO-Input für die korrekte Berechnung der "entfernten" Schrauben
        inf_input_path = os.path.join(inference_input_dir, f"{image_id}.txt")
        inf_labels = load_yolo_labels(inf_input_path)
        # Nur Klasse 0 (vorhandene) interessiert uns hier als Basis
        inf_input_data = inf_labels[inf_labels[:, 0] == 0][:, 1:5] if len(inf_labels) > 0 else np.empty((0, 4))

        pred_labels = load_yolo_labels(pred_path)
        
        # Trenne Prognosen nach Klasse
        pred_kept_bolts = pred_labels[pred_labels[:, 0] == 0][:, 1:5] if len(pred_labels) > 0 else np.empty((0, 4))
        pred_missing_bolts = pred_labels[pred_labels[:, 0] == 1][:, 1:5] if len(pred_labels) > 0 else np.empty((0, 4))

        # --- Zähle die neuen Metriken ---
        
        # 1. Entfernte Schrauben (Gelb)
        # Hier vergleichen wir YOLO-Input (inf_input_data) mit dem Output (pred_kept_bolts)
        # HINWEIS: Bei Scaling ist filter_input=False hardcoded, aber wir prüfen es trotzdem generisch
        count_removed = 0
        # Da filter_input=False erzwungen ist, ignorieren wir entfernte Schrauben im Dashboard,
        # es sei denn, sie fehlen wirklich im Output (was nicht passieren sollte).
        if inf_input_data.shape[0] > 0 and pred_kept_bolts.shape[0] > 0:
             dists = cdist(inf_input_data[:, :2], pred_kept_bolts[:, :2])
             min_dists_to_kept = np.min(dists, axis=1)
             # Nur zählen, wenn wirklich signifikant weg (Fehler im Skript?), sonst 0 annehmen
             real_removed = np.sum(min_dists_to_kept > dist_thresh)
             if real_removed > 0:
                 count_removed = int(real_removed)

        # 2. Evaluiere die Prognosen für fehlende Schrauben
        matched_gt_missing_indices = set()
        matched_pred_missing_indices = set()

        # Pass 1: Finde TPs (Grün) - Angleichung an evaluate.py (Iteriere über GT)
        if gt_missing_data.shape[0] > 0 and pred_missing_bolts.shape[0] > 0:
            dists = cdist(gt_missing_data[:,:2], pred_missing_bolts[:,:2])
            for i in range(gt_missing_data.shape[0]):
                best_pred_idx = np.argmin(dists[i, :])
                min_dist = dists[i, best_pred_idx]
                
                if min_dist < dist_thresh and best_pred_idx not in matched_pred_missing_indices:
                    matched_pred_missing_indices.add(best_pred_idx)
                    matched_gt_missing_indices.add(i)
                    
        count_tp_missing = len(matched_pred_missing_indices)
        
        # NEU: Pass 2 & 3: Finde fp_on_existing und fp_pure
        count_fp_on_existing = 0
        unmatched_pred_indices = [i for i in range(pred_missing_bolts.shape[0]) if i not in matched_pred_missing_indices]
        
        # `input_data` hier sind die GT class 0 Schrauben
        if len(unmatched_pred_indices) > 0 and input_data.shape[0] > 0:
            unmatched_preds = pred_missing_bolts[unmatched_pred_indices]
            dists = cdist(unmatched_preds[:, :2], input_data[:, :2])
            
            # KORREKTE LOGIK (wie in evaluate.py):
            # Prüfe für jede ungematchte Prognose, ob sie in der Nähe irgendeiner
            # vorhandenen GT-Schraube liegt.
            count_fp_on_existing = int(np.sum(np.min(dists, axis=1) < dist_thresh))

        # Übrige sind reine FPs (Rot)
        count_fp_pure = (pred_missing_bolts.shape[0] - count_tp_missing) - count_fp_on_existing

        # Pass 4: Übrige GTs sind FNs (Blau)
        count_fn_missing = gt_missing_data.shape[0] - len(matched_gt_missing_indices)

        # Verdeckte Fehler (YOLO FP maskiert Missing)
        # YOLO sagt "da ist was" (Class 0), Cluster behält es, aber GT sagt "da fehlt was" (Class 1).
        count_masking = 0
        if pred_kept_bolts.shape[0] > 0 and gt_missing_data.shape[0] > 0:
            dists = cdist(pred_kept_bolts[:, :2], gt_missing_data[:, :2])
            min_dists = np.min(dists, axis=1)
            count_masking = int(np.sum(min_dists < dist_thresh))

        # NEU: Falsche vorhandene Schrauben (FP Class 0), die übrig geblieben sind
        # Das sind Schrauben, die das Modell behalten hat, die aber nicht in GT Class 0 sind.
        count_fp_kept = 0
        if pred_kept_bolts.shape[0] > 0:
            if input_data.shape[0] > 0:
                dists = cdist(pred_kept_bolts[:, :2], input_data[:, :2])
                min_dists = np.min(dists, axis=1)
                count_fp_kept = int(np.sum(min_dists > dist_thresh))
            else:
                count_fp_kept = pred_kept_bolts.shape[0]

        all_results.append({
            "image_id": image_id,
            "tp_missing": count_tp_missing,
            "fp_pure": count_fp_pure,
            "fn_missing": count_fn_missing,
            "removed_fp": count_removed,
            "masking_fp": count_masking,
            "fp_on_existing": count_fp_on_existing,
            "fp_kept": count_fp_kept
        })
    return all_results

def compute_visualization_data(image_id, cfg):
    """Berechnet die Visualisierungsdaten für ein einzelnes Bild bei Bedarf."""
    val_input_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_input")
    val_gt_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_gt")
    
    run_name = cfg['inference'].get('run_name', 'default_run') # Kein _scaled mehr
    pred_dir = os.path.join(cfg['paths']['output_root'], "inference", run_name)
    inference_input_dir = cfg['paths']['inference_input_dir']

    # Lade Ground-Truth Daten für die finale Auswertung (Panel 3)
    gt_input_data = np.load(os.path.join(val_input_dir, f"{image_id}.npy"))
    gt_missing_data = np.load(os.path.join(val_gt_dir, f"{image_id}.npy"))

    # Lade die tatsächlichen Input-Daten, die für die Inferenz verwendet wurden (für Panel 1 & 2)
    inference_input_path = os.path.join(inference_input_dir, f"{image_id}.txt")
    all_inference_input_labels = load_yolo_labels(inference_input_path)
    if len(all_inference_input_labels) > 0:
        inference_input_labels = all_inference_input_labels[all_inference_input_labels[:, 0] == 0]
        yolo_missing_labels = all_inference_input_labels[all_inference_input_labels[:, 0] == 1]
    else:
        inference_input_labels = np.empty((0, 5))
        yolo_missing_labels = np.empty((0, 5))
    inference_input_data = inference_input_labels[:, 1:5]
    yolo_missing_data = yolo_missing_labels[:, 1:5]

    pred_labels = load_yolo_labels(os.path.join(pred_dir, f"{image_id}.txt"))
    pred_kept_bolts = pred_labels[pred_labels[:, 0] == 0][:, 1:5] if len(pred_labels) > 0 else np.empty((0, 4))
    pred_missing_bolts = pred_labels[pred_labels[:, 0] == 1][:, 1:5] if len(pred_labels) > 0 else np.empty((0, 4))

    # --- Sortiere alle Boxen in die neuen Kategorien für die Visualisierung ---
    dist_thresh = cfg['evaluation']['dist_threshold']
    
    # Kategorie: Behaltene vs. entfernte Input-Schrauben
    kept_input_bolts, removed_input_bolts = [], []
    # Bei Scaling ist filter_input=False, also sollten alle behalten werden.
    # Wir prüfen trotzdem gegen den Output.
    # Da wir wissen, dass filter_input=False ist, setzen wir kept=input.
    kept_input_bolts = list(inference_input_data)
    removed_input_bolts = [] 

    # Unterteile 'kept_input_bolts' in 'normal' und 'masking' (Verdeckte Fehler)
    normal_kept_bolts = []
    masking_kept_bolts = []
    
    if len(kept_input_bolts) > 0:
        kept_arr = np.array(kept_input_bolts)
        if gt_missing_data.shape[0] > 0:
             dists = cdist(kept_arr[:, :2], gt_missing_data[:, :2])
             min_dists = np.min(dists, axis=1)
             for i, d in enumerate(min_dists):
                 if d < dist_thresh:
                     masking_kept_bolts.append(kept_input_bolts[i])
                 else:
                     normal_kept_bolts.append(kept_input_bolts[i])
        else:
            normal_kept_bolts = list(kept_input_bolts)

    # Unterteile 'normal_kept_bolts' weiter in 'korrekt' und 'falsch verblieben' (FP Class 0)
    correct_kept_bolts = []
    fp_kept_bolts = []
    
    if len(normal_kept_bolts) > 0:
        normal_arr = np.array(normal_kept_bolts)
        if gt_input_data.shape[0] > 0:
            dists = cdist(normal_arr[:, :2], gt_input_data[:, :2])
            min_dists = np.min(dists, axis=1)
            for i, d in enumerate(min_dists):
                if d > dist_thresh:
                    fp_kept_bolts.append(normal_kept_bolts[i])
                else:
                    correct_kept_bolts.append(normal_kept_bolts[i])
        else:
            fp_kept_bolts = list(normal_kept_bolts)

    # Kategorien für fehlende Schrauben
    tp_missing, fp_on_existing, fp_pure, fn_missing = [], [], [], [] 
    matched_gt_missing_indices, matched_pred_missing_indices = set(), set()

    # Pass 1: Finde TPs (Grün)
    if gt_missing_data.shape[0] > 0 and pred_missing_bolts.shape[0] > 0:
        dists = cdist(gt_missing_data[:,:2], pred_missing_bolts[:,:2])
        for i in range(gt_missing_data.shape[0]):
            best_pred_idx = np.argmin(dists[i, :])
            min_dist = dists[i, best_pred_idx]
            
            if min_dist < dist_thresh and best_pred_idx not in matched_pred_missing_indices:
                tp_missing.append(pred_missing_bolts[best_pred_idx])
                matched_pred_missing_indices.add(best_pred_idx)
                matched_gt_missing_indices.add(i)

    # Pass 2 & 3: Teile die restlichen Prognosen in fp_pure und fp_on_existing auf
    for i in range(pred_missing_bolts.shape[0]):
        if i not in matched_pred_missing_indices:
            is_fp_on_existing = False
            if gt_input_data.shape[0] > 0:
                dists_to_gt_input = cdist(pred_missing_bolts[i:i+1, :2], gt_input_data[:, :2])
                if np.min(dists_to_gt_input) < dist_thresh:
                    fp_on_existing.append(pred_missing_bolts[i])
                    is_fp_on_existing = True
            
            if not is_fp_on_existing:
                fp_pure.append(pred_missing_bolts[i])

    # Pass 4: Übrige GTs sind FNs (Blau)
    for i in range(gt_missing_data.shape[0]):
        if i not in matched_gt_missing_indices:
            fn_missing.append(gt_missing_data[i])

    return {
        "input_data": inference_input_data, 
        "yolo_missing_data": yolo_missing_data,
        "pred_kept_bolts": pred_kept_bolts,
        "pred_missing_bolts": pred_missing_bolts,
        "correct_kept_bolts": np.array(correct_kept_bolts),
        "fp_kept_bolts": np.array(fp_kept_bolts),
        "masking_kept_bolts": np.array(masking_kept_bolts),
        "removed_input_bolts": np.array(removed_input_bolts),
        "tp_missing": np.array(tp_missing),
        "fp_pure": np.array(fp_pure),
        "fp_on_existing": np.array(fp_on_existing),
        "fn_missing": np.array(fn_missing),
    }

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("Cluster-Vervollständigung: Analyse-Dashboard (SCALED)")

# --- Sidebar für Steuerung ---
st.sidebar.header("Einstellungen")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Legende")
legend_html = """
<div style="display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px;">
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(255, 0, 255); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Cluster-Prognose (Skaliert)</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(0, 150, 255); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Vorhanden</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(0, 255, 255); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Fehlend</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(0, 255, 0); margin-right: 5px;"></div>
        <span>Korrekt</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(255, 255, 0); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Entfernt</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(255, 0, 0); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Falsch</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(255, 165, 0); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Falsch (auf Vorhandener)</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(0, 0, 255); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Übersehen/Verdeckt</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 12px; height: 12px; background-color: rgb(128, 0, 128); margin-right: 5px; border: 1px solid #ccc;"></div>
        <span>Falsch verblieben (FP Class 0)</span>
    </div>
</div>
"""
st.sidebar.markdown(legend_html, unsafe_allow_html=True)
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Cache leeren & Daten neu laden"):
    st.cache_data.clear()
    st.rerun()

# Lade alle Ergebnisse (wird gecached)
evaluation_results = load_and_evaluate_all_data()

@st.cache_resource
def load_models_and_configs():
    """Lädt Prototypen und Configs, die für die on-demand Visualisierung benötigt werden."""
    cfg = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))
    
    gnn_cfg_path = os.path.join(SCRIPT_DIR, '..', 'GNN', 'config.yaml')
    with open(gnn_cfg_path, 'r') as f:
        gnn_cfg = yaml.safe_load(f)
    image_folders = [gnn_cfg["paths"]["train_images"], gnn_cfg["paths"]["val_images"]]
    return cfg, image_folders

cfg, image_folders = load_models_and_configs()

# --- GESAMTSTATISTIK ANZEIGEN ---
total_tp = sum(r['tp_missing'] for r in evaluation_results)
total_fp_pure = sum(r['fp_pure'] for r in evaluation_results)
total_fp_on_existing = sum(r.get('fp_on_existing', 0) for r in evaluation_results)
total_fn = sum(r['fn_missing'] for r in evaluation_results)
total_fp_all = total_fp_pure + total_fp_on_existing

st.sidebar.markdown("### Gesamt-Statistik (Dashboard)")
st.sidebar.markdown(f"""
- **TP (Korrekt):** {total_tp}
- **FP (Gesamt):** {total_fp_all}
  - *Rein Falsch:* {total_fp_pure}
  - *Auf Vorhandener:* {total_fp_on_existing}
- **FN (Übersehen):** {total_fn}
- **Dist-Thresh:** {cfg['evaluation']['dist_threshold']}
""")
filter_option = st.sidebar.selectbox(
    "Filtere nach Ergebnis-Typ",
    [
        "Alle",
        "Perfekte Treffer (Nur TP)",
        "Fall 1: YOLO-Fehler entfernt (Gelb)",
        "Fall 2: Verdeckter Fehler (Blau)",
        "Fall 3: Falsch ergänzt (Rot)",
        "Fall 4: Übersehen (Blau)",
        "Fall 5: Vorhandene übersehen & falsch ergänzt",
        "Fall 6: Falsche vorhandene verblieben (FP Class 0)"
    ]
)
num_images = st.sidebar.slider("Anzahl der Beispiele", 1, 10, 3)

# Erklärungstexte für die Filter
explanations = {
    "Alle": "Zeigt alle Bilder ohne Filterung.",
    "Perfekte Treffer (Nur TP)": "YOLO hat eine Lücke gelassen. Das Cluster-Modell hat die fehlende Schraube an der richtigen Stelle ergänzt. Keine Fehler.",
    "Fall 1: YOLO-Fehler entfernt (Gelb)": "YOLO hat fälschlicherweise eine Schraube erkannt. Das Cluster-Modell hat erkannt, dass sie dort nicht hingehört und sie entfernt.",
    "Fall 2: Verdeckter Fehler (Blau)": "YOLO hat fälschlicherweise eine 'vorhandene' Schraube erkannt, wo eigentlich eine fehlt. Das Cluster-Modell hat die Position bestätigt, wodurch die Korrektur zur 'fehlenden' Schraube verhindert wurde.",
    "Fall 3: Falsch ergänzt (Rot)": "YOLO hat nichts erkannt, und das Cluster-Modell hat fälschlicherweise eine fehlende Schraube hinzugefügt, wo keine sein sollte.",
    "Fall 4: Übersehen (Blau)": "Eine fehlende Schraube wurde weder von YOLO noch vom Cluster-Modell erkannt.",
    "Fall 5: Vorhandene übersehen & falsch ergänzt": "YOLO hat eine tatsächlich vorhandene Schraube übersehen. Das Cluster-Modell hat an dieser Stelle fälschlicherweise eine 'fehlende' Schraube ergänzt.",
    "Fall 6: Falsche vorhandene verblieben (FP Class 0)": "Nach der Bereinigung sind noch 'vorhandene' Schrauben übrig, die laut Ground Truth dort nicht sein dürften (weder als vorhanden noch als fehlend)."
}

st.sidebar.info(f"**Info zum Filter:**\n\n{explanations[filter_option]}")

# --- Filtere die Ergebnisse basierend auf der Auswahl ---
filtered_results = []
for res in evaluation_results:
    # Logik für die Filter
    has_removed_fp = res["removed_fp"] > 0
    has_fp_pure = res["fp_pure"] > 0
    has_fn = res["fn_missing"] > 0
    has_masking = res["masking_fp"] > 0
    has_fp_on_existing = res.get("fp_on_existing", 0) > 0
    has_tp = res["tp_missing"] > 0
    has_fp_kept = res.get("fp_kept", 0) > 0
    is_perfect = has_tp and not has_removed_fp and not has_fp_pure and not has_fn and not has_masking

    if filter_option == "Alle":
        filtered_results.append(res)
    elif filter_option == "Perfekte Treffer (Nur TP)" and is_perfect:
        filtered_results.append(res)
    elif filter_option == "Fall 1: YOLO-Fehler entfernt (Gelb)" and has_removed_fp:
        filtered_results.append(res)
    elif filter_option == "Fall 2: Verdeckter Fehler (Blau)" and has_masking:
        filtered_results.append(res)
    elif filter_option == "Fall 3: Falsch ergänzt (Rot)" and has_fp_pure:
        filtered_results.append(res)
    elif filter_option == "Fall 4: Übersehen (Blau)" and has_fn:
        filtered_results.append(res)
    elif filter_option == "Fall 5: Vorhandene übersehen & falsch ergänzt" and has_fp_on_existing:
        filtered_results.append(res)
    elif filter_option == "Fall 6: Falsche vorhandene verblieben (FP Class 0)" and has_fp_kept:
        filtered_results.append(res)

if not filtered_results:
    st.warning("Keine Bilder für die ausgewählten Kriterien gefunden.")
else:
    # Wähle zufällige Indizes aus der gefilterten Liste
    num_to_sample = min(num_images, len(filtered_results))
    selected_results = random.sample(filtered_results, num_to_sample)

    # --- Hauptanzeige ---
    st.header(f"Zufällige Beispiele für: '{filter_option}'")
    
    for res in selected_results:
        image_id = res["image_id"]
        
        # --- ON-DEMAND BERECHNUNG FÜR VISUALISIERUNG ---
        with st.spinner(f"Visualisierung für {image_id} wird berechnet..."):
            vis_data = compute_visualization_data(image_id, cfg)

        st.divider()
        st.subheader(f"Bild: {image_id}")
        st.write(f"**Ergebnis:** "
                 f"Korrekt ergänzt (TP): **{res['tp_missing']}** | "
                 f"Entfernt: **{res['removed_fp']}** | "
                 f"Falsch ergänzt (FP): **{res['fp_pure']}** | "
                 f"Übersehen (FN): **{res['fn_missing']}** | "
                 f"Verdeckt: **{res['masking_fp']}** | "
                 f"Falsch auf Vorh.: **{res.get('fp_on_existing', 0)}** | "
                 f"Falsch verblieben: **{res.get('fp_kept', 0)}**")

        img_path = find_image_path(image_id, image_folders)
        if not img_path: continue
        image = cv2.imread(img_path)

        # Dynamische Liniendicke basierend auf Bildbreite für bessere Sichtbarkeit
        base_thickness = max(2, int(image.shape[1] / 720))

        # Erstelle die 3-Panel-Visualisierung on-the-fly
        COLOR_INPUT = (255, 150, 0)      # Hellblau
        COLOR_PROTOTYPE = (255, 0, 255)  # Magenta
        COLOR_YOLO_MISSING = (255, 255, 0) # Helles Türkis (Cyan)
        
        # Farben für Panel 3
        COLOR_TP_MISSING = (0, 255, 0)           # Grün
        COLOR_REMOVED_FP = (0, 255, 255)         # Gelb
        COLOR_MASKING_FP = (255, 0, 0)           # Dunkelblau (Verdeckter Fehler)
        COLOR_FP_PURE = (0, 0, 255)              # Rot
        COLOR_FP_ON_EXISTING = (0, 165, 255)     # Orange
        COLOR_FN_MISSING = (255, 0, 0)           # Blau (oder Rot/Orange je nach Wunsch, hier Blau wie angefordert "farblich hervorrufen")
        COLOR_FP_KEPT = (128, 0, 128)            # Dunkel Lila

        COLOR_ALIGN_LINE = (0, 255, 0)
        COLOR_TEXT = (255, 255, 255)
        
        # Panel 1: Input
        img1 = image.copy()
        draw_boxes(img1, vis_data["input_data"][:, :4], COLOR_INPUT, base_thickness + 1)
        if vis_data["yolo_missing_data"].shape[0] > 0:
            draw_boxes(img1, vis_data["yolo_missing_data"][:, :4], COLOR_YOLO_MISSING, base_thickness + 1)
        cv2.putText(img1, "1. Input (YOLO)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3)
        
        # Panel 2: Result (Raw Output)
        img2 = image.copy()
        # Zeige einfach das Ergebnis des Clusters (Behaltene + Fehlende)
        if vis_data["pred_kept_bolts"].shape[0] > 0:
            draw_boxes(img2, vis_data["pred_kept_bolts"], COLOR_INPUT, base_thickness + 1)
        if vis_data["pred_missing_bolts"].shape[0] > 0:
            draw_boxes(img2, vis_data["pred_missing_bolts"], COLOR_YOLO_MISSING, base_thickness + 1)
            
        cv2.putText(img2, "2. Cluster Output", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3)

        # Panel 3: Prognose
        img3 = image.copy()
        # Zeichne alle Boxen in der richtigen Reihenfolge für gute Sichtbarkeit
        # 1. Hintergrund: Vorhandene / Behaltene
        if vis_data["correct_kept_bolts"].shape[0] > 0:
            draw_boxes(img3, vis_data["correct_kept_bolts"], COLOR_INPUT, base_thickness + 1)
        if vis_data["fp_kept_bolts"].shape[0] > 0:
            color_fp_kept = COLOR_FP_KEPT if filter_option == "Fall 6: Falsche vorhandene verblieben (FP Class 0)" else COLOR_INPUT
            draw_boxes(img3, vis_data["fp_kept_bolts"], color_fp_kept, base_thickness + 1)
            
        if vis_data["removed_input_bolts"].shape[0] > 0:
            draw_boxes(img3, vis_data["removed_input_bolts"], COLOR_REMOVED_FP, base_thickness + 2)
        if vis_data["masking_kept_bolts"].shape[0] > 0:
            draw_boxes(img3, vis_data["masking_kept_bolts"], COLOR_MASKING_FP, base_thickness + 2)
        if vis_data["fn_missing"].shape[0] > 0:
            draw_boxes(img3, vis_data["fn_missing"], COLOR_FN_MISSING, base_thickness + 2)
            
        # 2. Vordergrund: Ergänzte Schrauben (damit man sie sieht!)
        if vis_data["fp_pure"].shape[0] > 0:
            draw_boxes(img3, vis_data["fp_pure"], COLOR_FP_PURE, base_thickness + 2)
        if vis_data["fp_on_existing"].shape[0] > 0:
            draw_boxes(img3, vis_data["fp_on_existing"], COLOR_FP_ON_EXISTING, base_thickness + 2)
        if vis_data["tp_missing"].shape[0] > 0:
            draw_boxes(img3, vis_data["tp_missing"], COLOR_TP_MISSING, base_thickness + 2)
        cv2.putText(img3, "3. Auswertung", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3)
        
        # Zeige die Bilder in drei Spalten an, um die Auflösung zu erhalten
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img1, channels="BGR", width='stretch')
        with col2:
            st.image(img2, channels="BGR", width='stretch')
        with col3:
            st.image(img3, channels="BGR", width='stretch')
