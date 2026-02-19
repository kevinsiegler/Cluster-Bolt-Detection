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

## 🛠 Installation

1.  **Python Umgebung:** Python 3.8+ wird empfohlen.
2.  **Abhängigkeiten installieren:**

```bash
pip install numpy scipy scikit-learn pyyaml tqdm matplotlib pandas
