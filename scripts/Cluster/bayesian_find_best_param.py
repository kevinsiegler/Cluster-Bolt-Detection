import os
import sys
import yaml
import numpy as np
import pickle
import optuna
from scipy.spatial.distance import cdist

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, load_yolo_labels, find_best_match, ensure_dir


class BayesianHyperparameterOptimizer:

    def __init__(self):

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.script_dir, "config.yaml")
        self.cfg = load_config(self.config_path)

        # --- Search Space (as specified) ---
        self.param_ranges = {
            'inlier_threshold': (0.01, 0.06),
            'acceptance_threshold': (0.1, 200.0),
            'missing_penalty': (0.1, 30.0),
            'outlier_penalty': (0.1, 3.0)
        }

        self.prototypes = self._load_prototypes()
        self.val_data = self._load_validation_data()

    # ---------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------

    def _load_prototypes(self):
        model_name = self.cfg['clustering'].get('model_name', 'prototypes')
        model_path = os.path.join(
            self.cfg['paths']['output_root'],
            self.cfg['paths']['model_dir'],
            f"{model_name}.pkl"
        )
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def _load_validation_data(self):
        input_dir = self.cfg['paths']['inference_input_dir']
        gt_missing_dir = os.path.join(
            self.cfg['paths']['output_root'],
            "preprocessing",
            "val_gt"
        )

        files = [f for f in os.listdir(gt_missing_dir) if f.endswith('.npy')]

        data = []
        for f in files:
            image_id = os.path.splitext(f)[0]

            yolo_path = os.path.join(input_dir, f"{image_id}.txt")
            yolo_labels = load_yolo_labels(yolo_path)

            gt_missing = np.load(os.path.join(gt_missing_dir, f))
            if gt_missing.ndim == 2 and gt_missing.shape[1] > 2:
                gt_missing = gt_missing[:, :2]

            data.append({
                'id': image_id,
                'yolo_labels': yolo_labels,
                'gt_missing': gt_missing
            })

        return data

    # ---------------------------------------------------------
    # INFERENCE + EVALUATION (EXACT LOGIC + YOLO FALLBACK)
    # ---------------------------------------------------------

    def run_inference_and_evaluate(self, params):

        tp = fp = fn = 0

        inlier_thresh = params['inlier_threshold']
        acc_thresh = params['acceptance_threshold']
        miss_pen = params['missing_penalty']
        out_pen = params['outlier_penalty']

        dist_thresh = self.cfg['evaluation']['dist_threshold']
        missing_detection_thresh = max(inlier_thresh * 2.0, 0.05)

        for sample in self.val_data:

            all_labels = sample['yolo_labels']

            pts_0 = all_labels[all_labels[:, 0] == 0] if len(all_labels) else np.empty((0, 5))
            pts_1 = all_labels[all_labels[:, 0] == 1] if len(all_labels) else np.empty((0, 5))

            match_pts_list = []
            if len(pts_0): match_pts_list.append(pts_0[:, 1:3])
            if len(pts_1): match_pts_list.append(pts_1[:, 1:3])

            input_pts = np.vstack(match_pts_list) if match_pts_list else np.empty((0, 2))
            predicted_missing_pts = []

            # -------------------------------------------------
            # YOLO FALLBACK (<=1 point → skip clustering)
            # -------------------------------------------------
            if len(input_pts) <= 1:
                if len(pts_1) > 0:
                    predicted_missing_pts = pts_1[:, 1:3].tolist()
                else:
                    predicted_missing_pts = []

            else:
                # --- Clustering ---
                best_proto, best_score = find_best_match(
                    input_pts,
                    self.prototypes,
                    inlier_thresh,
                    outlier_penalty=out_pen,
                    missing_penalty=miss_pen
                )

                if best_proto is not None and best_score < acc_thresh:

                    best_aligned_proto = best_proto['points'][:, :2]
                    input_pts_0 = pts_0[:, 1:3] if len(pts_0) else np.empty((0, 2))

                    dists = cdist(best_aligned_proto, input_pts_0) \
                        if len(input_pts_0) else np.full((len(best_aligned_proto), 0), np.inf)

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

            # ---------------- Evaluation ----------------

            pred_pts = np.array(predicted_missing_pts)
            gt_pts = sample['gt_missing']

            n_gt = len(gt_pts)
            n_pred = len(pred_pts)

            if n_gt == 0 and n_pred == 0:
                continue

            if n_gt == 0:
                fp += n_pred
                continue

            if n_pred == 0:
                fn += n_gt
                continue

            dists_eval = cdist(gt_pts, pred_pts)
            matched_gt = set()
            matched_pred = set()

            for i in range(n_gt):
                best_match_idx = np.argmin(dists_eval[i])
                min_dist = dists_eval[i][best_match_idx]

                if min_dist < dist_thresh and best_match_idx not in matched_pred:
                    matched_pred.add(best_match_idx)
                    matched_gt.add(i)
                    tp += 1

            fp += (n_pred - len(matched_pred))
            fn += (n_gt - len(matched_gt))

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return recall, precision, f1

    # ---------------------------------------------------------
    # OPTUNA OBJECTIVE (Recall leicht wichtiger)
    # ---------------------------------------------------------

    def objective(self, trial):

        params = {
            'inlier_threshold': trial.suggest_float('inlier_threshold', 0.01, 0.06),
            'acceptance_threshold': trial.suggest_float('acceptance_threshold', 0.1, 200.0, log=True),
            'missing_penalty': trial.suggest_float('missing_penalty', 0.1, 30.0, log=True),
            'outlier_penalty': trial.suggest_float('outlier_penalty', 0.1, 3.0, log=True)
        }

        recall, precision, f1 = self.run_inference_and_evaluate(params)

        score = (
            3.0 * recall +
            2.5 * precision +
            1.0 * f1
        )

        return score

    # ---------------------------------------------------------

    def optimize(self, n_trials=350):

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(self.objective, n_trials=n_trials)

        best_params = study.best_params
        recall, precision, f1 = self.run_inference_and_evaluate(best_params)

        self.save_results(best_params, recall, precision, f1)

    # ---------------------------------------------------------

    def save_results(self, params, recall, precision, f1):

        output_dir = os.path.join(
            self.cfg['paths']['output_root'],
            "best_parameter_settings"
        )
        ensure_dir(output_dir)

        with open(os.path.join(output_dir, "best_params.yaml"), "w") as f:
            yaml.dump(params, f)

        with open(os.path.join(output_dir, "optimization_report.txt"), "w") as f:
            f.write("=== BEST PARAMETER SETTINGS (Bayesian Optimization) ===\n\n")
            yaml.dump(params, f)
            f.write("\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")


if __name__ == "__main__":
    opt = BayesianHyperparameterOptimizer()
    opt.optimize(n_trials=350)