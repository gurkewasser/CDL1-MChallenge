# CDL1-MChallenge – Sensor-based Activity Recognition

Dieses Projekt befasst sich mit der Erkennung menschlicher Aktivitäten auf Basis von Sensordaten (z. B. Joggen, Sitzen etc.). Es beinhaltet sowohl klassische Machine-Learning-Ansätze als auch Deep-Learning-Modelle zur Klassifikation von segmentierten Sensor-Zeitreihendaten.

## 📁 Projektstruktur

```
CDL1-MChallenge
├── README.md
├── data
├── download_link_data.txt
├── images
│   └── image_standarized.png
├── main.py
├── notebooks
│   ├── Overview_Challenge.ipynb
│   ├── Plots
│   │   └── NDL_Modelvergleich.png
│   ├── data
│   │   └── joggen_8-2025_04_08_10-03-31.zip
│   ├── deep_learning_models.ipynb
│   └── non_deep_learning_models.ipynb
├── requirements.txt
└── src
    ├── config.yaml
    ├── config_lstm.yaml
    ├── config_mlp.yaml
    ├── grid_search_cnn.py
    ├── grid_search_lstm.py
    ├── grid_search_mlp.py
    ├── methods.py
    ├── train_cnn.py
    ├── train_lstm.py
    ├── train_mlp.py
    └── utils.py
```

## Datendownload

Die Trainingsdaten können unter folgendem Link heruntergeladen werden:

[Download via SWITCH Filesender](https://filesender.switch.ch/filesender2/?s=download&token=b371fe3a-597e-4b03-a7da-808be2a050f4)

## Einstiegspunkt

Die Datei `main.py` ist die Pipeline, die alle Daten vorbereitet und aufbereitet, um sie für das Training der klassischen und Deep-Learning-Modelle nutzbar zu machen. Um das Skript auszuführen, muss man einen Ordner `data` und darin einen Unterordner `raw` erstellen und die heruntergeladene **data.zip-Datei** darin speichern. Das Skript bereitet die Daten nach der Ausführung automatisch vor.

- `Overview_Challenge.ipynb`: Überblick über die Challenge und die verwendeten Daten und Vorbereitung der Daten
- `non_deep_learning_models.ipynb`: Klassische Modelle
- `deep_learning_models.ipynb`: CNN, LSTM, MLP

## Konfiguration

Die Parameter für die Modelle werden in YAML-Dateien im `src/` Verzeichnis definiert:

- `config.yaml`, `config_lstm.yaml`, `config_mlp.yaml`

## Modelltraining

In `src/` befinden sich Trainings- und GridSearch-Skripte:

- `train_cnn.py`, `train_lstm.py`, `train_mlp.py`
- `grid_search_cnn.py`, `grid_search_lstm.py`, `grid_search_mlp.py`

## Abhängigkeiten

Alle nötigen Python-Pakete sind in der Datei `requirements.txt` aufgeführt. Installation via:

```bash
pip install -r requirements.txt
```

## Visualisierungen

Im Ordner `images/` befinden sich Visualisierungen zur Datenverarbeitung und zum Modellvergleich.
