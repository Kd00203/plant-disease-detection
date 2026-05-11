# 🌿 Plant Disease Detection System

A deep learning–powered web application that detects plant diseases from leaf images using a fine-tuned **EfficientNetB0** CNN model, served via a **Flask REST API**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.3-green?logo=flask)
![Accuracy](https://img.shields.io/badge/Accuracy-91%25-brightgreen)

---

## 📌 Features

- **End-to-end CNN pipeline** — data ingestion, augmentation, training, evaluation, and inference
- **Transfer learning** with EfficientNetB0 pretrained on ImageNet + fine-tuning
- **91% classification accuracy** on the PlantVillage dataset (38 classes)
- **Flask REST API** (`/api/predict`) for real-time image inference
- **Web UI** — drag & drop leaf image upload with instant results
- Identifies both the **plant type** and **disease** from a single image

---

## 🗂 Project Structure

```
plant-disease-detection/
│
├── model/
│   ├── train.py              # Training pipeline (augmentation + fine-tuning)
│   ├── predict.py            # Inference utility
│   └── saved_model/          # Saved Keras model (after training)
│       ├── plant_disease_model.keras
│       └── class_names.json
│
├── api/
│   └── app.py                # Flask REST API
│
├── templates/
│   └── index.html            # Web UI
│
├── static/uploads/           # Temp image storage
├── data/
│   └── train/                # PlantVillage dataset (add here)
│
├── notebooks/
│   └── EDA_and_Evaluation.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle and place it at:
```
data/train/<class_name>/image.jpg
```

---

## 🚀 Usage

### Train the model
```bash
python model/train.py
```
Outputs:
- `model/saved_model/plant_disease_model.keras`
- `model/saved_model/class_names.json`
- `model/training_curves.png`

### Run the Flask API
```bash
python api/app.py
```
Visit `http://localhost:5000` to use the web UI.

### API — predict endpoint
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@leaf.jpg"
```

**Response:**
```json
{
  "success": true,
  "plant": "Tomato",
  "disease": "Early Blight",
  "confidence": 91.42,
  "is_healthy": false,
  "top5": [...]
}
```

---

## 📊 Model Performance

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 91%    |
| Dataset     | PlantVillage (5,000+ images) |
| Classes     | 38     |
| Architecture| EfficientNetB0 (fine-tuned) |

---

## 🛠 Tech Stack

| Component       | Technology                      |
|-----------------|---------------------------------|
| Deep Learning   | TensorFlow 2.x / Keras          |
| Image Processing| OpenCV, PIL                     |
| Backend API     | Flask                           |
| Frontend        | HTML / CSS / Vanilla JS         |
| Training        | Google Colab / GPU              |

---

## 📝 License

MIT License — feel free to use, modify, and distribute.

---

## 👩‍💻 Author

**Kalyani Deshmane** — AI/ML Engineer  
📧 kalyanideshmane3@gmail.com | [GitHub](https://github.com/Kd00203)
