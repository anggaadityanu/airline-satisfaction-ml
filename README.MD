# ✈️ Airline Passenger Satisfaction Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Streamlit-1.58-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/XGBoost-ML-success?style=for-the-badge">
</p>

## 📌 Project Description

Airline Passenger Satisfaction Prediction adalah proyek Machine Learning yang bertujuan untuk memprediksi tingkat kepuasan penumpang maskapai berdasarkan berbagai faktor pelayanan penerbangan.

Proyek ini membandingkan performa empat algoritma Machine Learning:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Model terbaik kemudian diimplementasikan ke dalam aplikasi web menggunakan Streamlit sehingga pengguna dapat melakukan prediksi secara langsung.

---

# 📂 Project Structure

```
Airline Satisfaction ML/
│
├── dataset/
│   └── Airline_customer_satisfaction.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── label_encoders.pkl
│   └── standard_scaler.pkl
│
├── notebook/
│   └── airline_satisfaction.ipynb
│
├── streamlit/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

---

# 🎯 Objectives

- Melakukan preprocessing dataset.
- Membersihkan data.
- Melakukan encoding data kategorikal.
- Feature Scaling.
- Melatih beberapa model Machine Learning.
- Membandingkan performa model.
- Menentukan model terbaik.
- Deploy model menggunakan Streamlit.

---

# 📊 Dataset

Dataset yang digunakan adalah Airline Customer Satisfaction Dataset.

Jumlah data

```
129.880
```

Jumlah feature

```
21
```

Target

```
Satisfaction
```

---

# 🛠 Technology

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- Joblib
- Streamlit

---

# ⚙ Installation

Clone repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
```

Masuk ke folder project

```bash
cd airline-satisfaction-ml
```

---

## Membuat Virtual Environment

Windows

```bash
python -m venv venv
```

Aktifkan

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

atau

```bash
pip install pandas numpy matplotlib scikit-learn xgboost streamlit joblib
```

---

# ▶ Menjalankan Notebook

Masuk folder notebook

```bash
cd notebook
```

Buka menggunakan VS Code atau Jupyter Notebook.

Jalankan seluruh cell secara berurutan menggunakan **Run All**.

Notebook akan melakukan:

- Import Library
- Load Dataset
- Exploratory Data Analysis
- Data Cleaning
- Encoding
- Feature Scaling
- Split Dataset
- Training Model
- Evaluation
- Cross Validation
- Save Model (.pkl)

---

# 🚀 Menjalankan Streamlit

Masuk ke folder streamlit

```bash
cd streamlit
```

Jalankan

```bash
streamlit run app.py
```

Browser akan terbuka di

```
http://localhost:8501
```

---

# 🧠 Machine Learning Workflow

```
Dataset

↓

Preprocessing

↓

Cleaning

↓

EDA

↓

Encoding

↓

Feature Scaling

↓

Train Test Split

↓

Training Model

↓

Evaluation

↓

Cross Validation

↓

Save Model

↓

Deploy Streamlit
```

---

# 🤖 Algorithms

| Model | Description |
|---------|------------|
| Logistic Regression | Linear Classification |
| Decision Tree | Tree Based Classification |
| Random Forest | Ensemble Learning |
| XGBoost | Gradient Boosting |

---

# 📈 Evaluation Metrics

Model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Cross Validation

---

# 🏆 Best Model

Model terbaik berdasarkan hasil evaluasi adalah:

| Model | Accuracy |
|---------|-----------|
| XGBoost | **95.80%** |

---

# 💻 Streamlit Features

Aplikasi menyediakan fitur:

- Input data penumpang
- Prediksi menggunakan 4 model
- Confidence Score
- Ringkasan data input
- Perbandingan performa model
- Grafik Accuracy
- Grafik Precision
- Grafik Recall
- Grafik F1 Score

---

# 📷 Application Preview

Tambahkan screenshot aplikasi di sini.

```
images/dashboard.png
```

---

# 📄 Requirements

Contoh isi `requirements.txt`

```text
pandas
numpy
matplotlib
scikit-learn
streamlit
xgboost
joblib
```

Generate otomatis

```bash
pip freeze > requirements.txt
```

---

# 👨‍💻 Author

**Angga Aditya**

Universitas Esa Unggul

Machine Learning Project

2026

---

# ⭐ Support

Apabila repository ini bermanfaat, jangan lupa memberikan ⭐ pada repository GitHub ini.