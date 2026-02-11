# Autoencoder Spatial Validator


Kurzanleitung


1. Python venv erstellen: `python -m venv .venv` und aktivieren
2. Installieren: `pip install -r requirements.txt`
3. Pfade in `config.py` prüfen (BASE_DIR)
4. Train: `python train_autoencoder.py`
5. Thresholds: `python compute_thresholds.py`
6. Validieren: `python validate_yolo.py --yolo_preds "C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\YOLO\infer\evaluations_w_confidence_txt_data\infer_train_30_epoch_conf(0.01)" --images "C:\Users\Kevin\Clustererkennung\bolt_detection\dataset\images\val"`
7. Visual: `python visualize_results.py --image <imagepath> --txt <prediction_txt>`


Hinweise
- GRID_SIZE: 64 ist ein guter Startpunkt. Kleinere Grids = schneller, gröbere Lokalisation.
- Die Schwellenwerte (`thresholds.json`) werden aus Val-Labels berechnet.
- Für jeden Fahrzeugtyp kann ein eigenes Modell sinnvoll sein.