# Bolt Cluster Completion

Ein schlankes, zuverlässiges und unüberwachtes System zur Erkennung fehlender Schraubenpositionen basierend auf geometrischem Clustering.

## 🎯 Projektübersicht und Zielsetzung

Das Hauptziel dieses Projekts ist die Entwicklung eines reproduzierbaren End-to-End-Workflows, der aus manuell gelabelten YOLO-Daten (Train/Val) wiederkehrende Schrauben-Layouts identifiziert und daraus fehlende Schraubenpositionen bei partiellen Beobachtungen ergänzt.

**Kernprinzip:** Anstatt ein weiteres Deep-Learning-Modell für die Erkennung "fehlender" Schrauben zu trainieren, nutzen wir die strukturelle Natur von Schrauben-Layouts. Schrauben folgen oft wiederkehrenden Mustern. Dieses System lernt diese Muster (Prototypen) und verwendet sie, um unvollständige Beobachtungen zu vervollständigen.

**Workflow-Phasen:**
1.  **Preprocessing:** Aufbereitung der YOLO-Labels in normalisierte Punktmengen.
2.  **Clustering / Training:** Identifikation und Speicherung von Prototypen wiederkehrender Schrauben-Layouts aus den Trainingsdaten.
3.  **Inferenz:** Vorhersage fehlender Schraubenpositionen basierend auf partiellen Beobachtungen und den gelernten Prototypen.
4.  **Evaluation:** Quantitative Bewertung der Vorhersagegenauigkeit.

## ⚙️ Funktionsweise im Detail

### 1. Training (Clustering)
In dieser Phase lernt das System die typischen Schrauben-Anordnungen.
- **Datenladen:** Alle Ground-Truth-Labels des Trainingsdatensatzes werden geladen.
- **Normalisierung:** Die Koordinaten werden auf [0, 1] normalisiert, um unabhängig von der Bildgröße zu sein.
- **K-Means Clustering:** Der Algorithmus gruppiert ähnliche Punktmuster. Die Anzahl der Cluster (`n_clusters`) bestimmt, wie viele verschiedene "Prototypen" gelernt werden. Der `random_state` sorgt dabei für reproduzierbare Ergebnisse.
- **Pruning:** Sehr ähnliche Prototypen werden verschmolzen, um Redundanz zu vermeiden (`pruning_threshold`).
- **Speichern:** Die resultierenden Prototypen (Cluster-Zentren) werden als Modell (`.pkl`) gespeichert.

### 2. Inferenz (Vorhersage)
Hier werden fehlende Schrauben in neuen, unvollständigen Daten ergänzt.
- **Matching:** Für ein Eingabebild (z.B. YOLO-Vorhersagen) sucht das System den am besten passenden Prototypen.
- **Ausrichtung:** Der Prototyp wird über das Eingabemuster gelegt. Dabei wird er verschoben und optional skaliert (`allow_scaling`), um die beste Überdeckung zu erreichen.
- **Filterung:** Nur wenn der mittlere Abstand zwischen den Punkten klein genug ist (`match_threshold`), gilt der Prototyp als passend.
- **Ergänzung:** Punkte, die im Prototyp existieren, aber im Eingabebild fehlen, werden als "fehlende Schrauben" hinzugefügt.

### 3. Evaluation
Die Qualität der Ergänzungen wird quantitativ gemessen.
- **Vergleich:** Die vorhergesagten (ergänzten) Punkte werden mit den tatsächlichen Ground-Truth-Daten des Validierungssets verglichen.
- **Metriken:**
    - **Precision:** Wie viele der hinzugefügten Schrauben waren tatsächlich korrekt? (Vermeidung von "Geister-Schrauben")
    - **Recall:** Wie viele der tatsächlich fehlenden Schrauben wurden gefunden?
    - **F1-Score:** Das harmonische Mittel aus Precision und Recall.

## 🛠 Installation

1.  **Python Umgebung:** Python 3.8+ wird empfohlen.
2.  **Abhängigkeiten installieren:**

```bash
pip install numpy scipy scikit-learn pyyaml tqdm matplotlib pandas
