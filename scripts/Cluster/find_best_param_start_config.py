import os
import sys
import yaml
import numpy as np
import pickle
from scipy.spatial.distance import cdist
import optuna

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, load_yolo_labels, find_best_match, ensure_dir


class BayesianOptimizer:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg = load_config(os.path.join(self.script_dir, "config.yaml"))
        self.prototypes = self._load_prototypes()
        self.val_data = self._load_validation_data()

        self.inlier_threshold = self.cfg['inference'].get('inlier_threshold', 0.04)
        self.max_runs = 100

    # -------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------
    def _load_prototypes(self):
        model_name = self.cfg['clustering'].get('model_name', 'prototypes')
        model_path = os.path.join(self.cfg['paths']['output_root'], self.cfg['paths']['model_dir'], f"{model_name}.pkl")
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def _load_validation_data(self):
        input_dir = self.cfg['paths']['inference_input_dir']
        gt_missing_dir = os.path.join(self.cfg['paths']['output_root'], "preprocessing", "val_gt")
        
        # FIX: Iteriere über input_dir (Subset) statt gt_missing_dir (Full Set)
        # Dies stellt sicher, dass wir nur Dateien bewerten, für die YOLO-Input existiert (wie evaluate.py).
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        
        data = []
        for f in input_files:
            image_id = os.path.splitext(f)[0]
            gt_path = os.path.join(gt_missing_dir, f"{image_id}.npy")
            if not os.path.exists(gt_path): continue
            
            yolo_labels = load_yolo_labels(os.path.join(input_dir, f))
            gt_missing = np.load(gt_path)
            if gt_missing.ndim == 2 and gt_missing.shape[1] > 2:
                gt_missing = gt_missing[:, :2]
            data.append({'id': image_id, 'yolo_labels': yolo_labels, 'gt_missing': gt_missing})
        return data

    # -------------------------------------------------------------
    # ZIELFUNKTION FÜR OPTUNA
    # -------------------------------------------------------------
    def objective(self, trial):
        """Diese Funktion wird von Optuna für jeden Versuch aufgerufen."""
        params = {
            # Log-Skala ist effizient für Bereiche, die mehrere Größenordnungen umfassen.
            # Optuna wählt hier intelligent Werte zwischen 3 und 200.
            "acceptance_threshold": trial.suggest_float("acceptance_threshold", 3.0, 200.0, log=True),
            # Bereich [0.01, 15.0] wie gewünscht.
            "missing_penalty": trial.suggest_float("missing_penalty", 0.01, 15.0, log=True),
            # Bereich [0.1, 3.0] wie gewünscht.
            "outlier_penalty": trial.suggest_float("outlier_penalty", 0.1, 3.0, log=True)
        }

        tp = fp = fn = 0
        acc_thresh = params['acceptance_threshold']
        miss_pen = params['missing_penalty']
        out_pen = params['outlier_penalty']

        dist_thresh = self.cfg['evaluation']['dist_threshold']
        missing_detection_thresh = max(self.inlier_threshold * 2.0, 0.05)

        for sample in self.val_data:
            all_labels = sample['yolo_labels']
            pts_0 = all_labels[all_labels[:, 0] == 0] if len(all_labels) else np.empty((0, 5))
            pts_1 = all_labels[all_labels[:, 0] == 1] if len(all_labels) else np.empty((0, 5))

            match_pts_list = []
            if len(pts_0): match_pts_list.append(pts_0[:, 1:3])
            if len(pts_1): match_pts_list.append(pts_1[:, 1:3])
            input_pts = np.vstack(match_pts_list) if match_pts_list else np.empty((0, 2))

            predicted_missing_pts = []
            if len(input_pts) <= 1:
                predicted_missing_pts = pts_1[:, 1:3] if len(pts_1) else []
            else:
                best_proto, best_score = find_best_match(input_pts, self.prototypes,
                                                        self.inlier_threshold,
                                                        outlier_penalty=out_pen, missing_penalty=miss_pen)
                if best_proto is not None and best_score < acc_thresh:
                    best_aligned_proto = best_proto['points'][:, :2]
                    input_pts_0 = pts_0[:, 1:3] if len(pts_0) else np.empty((0, 2))
                    dists = cdist(best_aligned_proto, input_pts_0) if len(input_pts_0) else np.full((len(best_aligned_proto), 0), np.inf)
                    dists_copy = dists.copy()
                    matched_proto_indices = set()
                    while True:
                        if dists_copy.size == 0 or np.all(np.isinf(dists_copy)):
                            break
                        min_idx = np.unravel_index(np.argmin(dists_copy), dists_copy.shape)
                        min_dist = dists_copy[min_idx]
                        if min_dist > missing_detection_thresh:
                            break
                        p_idx, i_idx = min_idx
                        matched_proto_indices.add(p_idx)
                        dists_copy[p_idx, :] = np.inf
                        dists_copy[:, i_idx] = np.inf
                    for idx in range(len(best_aligned_proto)):
                        if idx not in matched_proto_indices:
                            pt = best_aligned_proto[idx]
                            if 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1:
                                predicted_missing_pts.append(pt)

            pred_pts = np.array(predicted_missing_pts)
            # Ensure 2D shape for cdist, even if empty
            if pred_pts.ndim != 2:
                pred_pts = pred_pts.reshape(-1, 2)
                
            gt_pts = sample['gt_missing']
            n_gt, n_pred = len(gt_pts), len(pred_pts)
            if n_gt == 0 and n_pred == 0:
                continue
            if n_gt == 0:
                fp += n_pred
                continue
            if n_pred == 0:
                fn += n_gt
                continue
            dists_eval = cdist(gt_pts, pred_pts)
            matched_gt, matched_pred = set(), set()
            for i in range(n_gt):
                best_match_idx = np.argmin(dists_eval[i])
                if dists_eval[i][best_match_idx] < dist_thresh and best_match_idx not in matched_pred:
                    matched_pred.add(best_match_idx)
                    matched_gt.add(i)
                    tp += 1
            fp += (n_pred - len(matched_pred))
            fn += (n_gt - len(matched_gt))

        recall = tp / (tp + fn) if (tp + fn) else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0

        # Speichere zusätzliche Metriken für die Analyse
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)

        # Optuna maximiert den zurückgegebenen Wert
        return f1

    # -------------------------------------------------------------
    # OPTIMIERUNG (BAYESIAN)
    # -------------------------------------------------------------
    def optimize(self):
        print(f"Starte Bayes'sche Optimierung mit {self.max_runs} Versuchen.")
        # Der TPE-Sampler ist der Kern der intelligenten Suche
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Starte die Optimierung
        study.optimize(self.objective, n_trials=self.max_runs, callbacks=[self.print_callback])

        # Beste Ergebnisse nach Abschluss der Studie abrufen
        best_trial = study.best_trial
        self.save_results(
            params=best_trial.params,
            f1=best_trial.value,
            recall=best_trial.user_attrs["recall"],
            precision=best_trial.user_attrs["precision"]
        )

    def print_callback(self, study, trial):
        """Callback-Funktion, um den Fortschritt in der gewünschten Form auszugeben."""
        print(
            f"[{trial.number + 1}/{self.max_runs}] "
            f"F1: {trial.value:.4f} | "
            f"Best F1: {study.best_value:.4f} | "
            f"Params: {trial.params}"
        )

    # -------------------------------------------------------------
    # SPEICHERN
    # -------------------------------------------------------------
    def save_results(self, params, f1, recall, precision):
        output_dir = os.path.join(self.cfg['paths']['output_root'], "best_parameter_settings")
        ensure_dir(output_dir)
        metrics = {"recall": recall, "precision": precision, "f1": f1}
        with open(os.path.join(output_dir, "best_params.yaml"), "w") as f:
            yaml.dump(params, f)
        with open(os.path.join(output_dir, "optimization_report.txt"), "w") as f:
            f.write("=== BEST PARAMETER SETTINGS (Bayesian Optimization) ===\n\n")
            yaml.dump(params, f)
            f.write("\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n")


if __name__ == "__main__":
    optimizer = BayesianOptimizer()
    optimizer.optimize()