# Insurance Premium Prediction using Machine Learning

## 📌 Project Overview
This project builds an **end-to-end machine learning pipeline** to predict **health insurance premiums** based on demographic, lifestyle, and medical attributes of individuals.

The solution is designed in an **industry-ready manner**, covering:
- Data ingestion from a SQLite database
- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Model experimentation & selection
- Production-ready training and prediction scripts

---

## 🎯 Business Objective
To predict **insurance premium charges** accurately so that insurance providers can:
- Assess individual risk
- Price policies fairly
- Automate premium calculation at scale

---

## 📂 Dataset Information
- **Source:** SQLite database (`regression.db`)
- **Table Used:** `insurance_prediction`
- **Total Records:** ~1,000,000

### Features
- `age` – Age of the individual  
- `gender` – Male/Female  
- `bmi` – Body Mass Index  
- `children` – Number of children  
- `smoker` – Smoking status  
- `region` – Residential region  
- `medical_history` – Existing medical conditions  
- `family_medical_history` – Family medical background  
- `exercise_frequency` – Exercise habits  
- `occupation` – Type of occupation  
- `coverage_level` – Insurance coverage level  

**Target Variable**
- `charges` – Insurance premium amount

---

## 🧠 Data Splitting Strategy
To simulate real-world deployment, the data was split **sequentially**:

| Purpose | Records |
|-------|--------|
| Training | First 700,000 |
| Evaluation | Next 200,000 |
| Production | Remaining records |

This avoids data leakage and mirrors real production scenarios.

---

## 📊 Exploratory Data Analysis (EDA)
EDA was performed on a **random sample of 100,000 records** for efficiency.

### Key Insights
- Insurance charges are **right-skewed**
- **Smokers pay significantly higher premiums**
- **Age and BMI** positively correlate with charges
- Medical history features contain high missing values (handled in preprocessing)

---

## ⚙️ Data Preprocessing
- **Numerical features:** Median imputation + StandardScaler  
- **Categorical features:** Missing values treated as `"Unknown"` + OneHotEncoding  
- Preprocessing implemented using **Scikit-learn Pipelines**
- Same preprocessing reused during **training and prediction**

---

## 🧪 Models Evaluated
The following models were experimented with:

| Model | Purpose |
|-----|--------|
| Ridge Regression | Linear baseline |
| SGDRegressor | Scalable linear baseline |
| Random Forest Regressor | Final model |

### ✅ Final Model: Random Forest Regressor
Chosen due to:
- Superior performance
- Ability to capture non-linear relationships
- Numerical stability on large tabular datasets

---

## 📈 Model Performance (Evaluation Set)
- **R²:** ~0.99  
- **MAE:** Low relative error  
- **RMSE:** Significantly better than linear baselines  

---

## 🔍 Feature Importance
Top influential features:
- Smoking status
- BMI
- Age
- Coverage level
- Medical history indicators

These align well with real-world insurance risk assessment logic.


---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Train the model
cd src
python train.py

### 3️⃣ Generate predictions
python predict.py

Predictions will be saved as a CSV file.

## 🏁 Conclusion

This project demonstrates a complete, industry-grade machine learning workflow, from data ingestion to deployment-ready prediction. The solution is scalable, reproducible, and aligned with real-world insurance pricing use cases.

## 👤 Author
### Ranjana Patidar
Senior Software Analyst | iOS Developer transitioning into AI & Data Science