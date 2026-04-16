# Customer Churn Prediction 🚀

## 📌 Project Overview

This project predicts whether a customer is likely to churn using machine learning techniques. The goal is to help businesses identify high-risk customers and take proactive retention actions.

---

## 🧠 Approach

* Performed data preprocessing using pipelines
* Handled missing values and categorical encoding
* Applied **SMOTE** to handle class imbalance
* Used **Stratified K-Fold Cross Validation** for robust evaluation
* Compared multiple models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Neural Network (MLP)
* Tuned hyperparameters using **GridSearchCV**
* Selected final model based on **recall and F1-score**

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Streamlit (for deployment)

---

## 📊 Model Performance

* Accuracy: ~0.73
* Recall: High (focused on catching churn)
* F1 Score: Balanced performance

---

## 💡 Key Insights

* Features like **Contract type, Tenure, and Monthly Charges** significantly impact churn
* Class imbalance was handled effectively using SMOTE
* Cross-validation ensured reliable model selection

---

## 🚀 Streamlit App

The project includes an interactive web app built using Streamlit where users can input customer details and get churn predictions.

---




---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

* `app.py` → Streamlit application
* `model.pkl` → Trained pipeline
* `notebook/` → Jupyter notebook with full workflow

---

## 🎯 Future Improvements

* Feature selection for optimization
* Try advanced models (XGBoost, LightGBM)
* Deploy on cloud (Streamlit Cloud / Render)

---

## 👤 Author

AMARPAL SINGH DUTTA
