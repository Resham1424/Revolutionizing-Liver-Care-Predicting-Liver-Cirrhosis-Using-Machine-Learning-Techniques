# Revolutionizing-Liver-Care-Predicting-Liver-Cirrhosis-Using-Machine-Learning-Techniques
"This project predicts liver cirrhosis using machine learning. It involves data preprocessing, EDA, model building, and evaluation to assist in early diagnosis and improve patient care."
# 🧬 Liver Cirrhosis Prediction

A machine learning project that predicts liver cirrhosis using clinical data from patients. It uses a Random Forest classifier and includes preprocessing with a standard scaler.

---

## 📁 Project Files

- `liver_cirrhosis.csv`: Sample dataset (age, bilirubin, enzymes, proteins, etc.)
- `rf_acc_68.pkl`: Trained Random Forest model
- `normalizer.pkl`: Scikit-learn StandardScaler used for feature scaling
- `requirements.txt`: Python dependencies
- `Liver_Cirrhosis_Prediction_Report.pdf`: Full project report
- `README.md`: Project overview and usage instructions

---

## 🚀 How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
import joblib

model = joblib.load("rf_acc_68.pkl")
scaler = joblib.load("normalizer.pkl")

sample_input = [[45, 1.2, 0.4, 230, 45, 50, 6.5, 3.3, 1.1]]
scaled = scaler.transform(sample_input)
prediction = model.predict(scaled)
print("Prediction:", prediction)
Target:

1 = Cirrhosis Present

0 = Not Present

📄 Report
See Liver_Cirrhosis_Prediction_Report.pdf for detailed explanation of the project.

