# Project: Credit Card Fraud Detection

**Author**: Maura E. Ramirez-Quezada  

**Description**: This project tackles the challenge of detecting fraudulent credit card transactions using supervised machine learning techniques. Given the highly imbalanced nature of the dataset — where fraud represents only 0.17% of all transactions — we focus not just on accuracy but on more informative metrics like precision, recall, and the Area Under the Precision-Recall Curve (AUPRC).

Starting with a simple logistic regression as baseline, we progressively improve model performance through threshold tuning and more complex classifiers such as Random Forest and XGBoost. We analyze model predictions, adjust decision boundaries, and use precision-recall curves to evaluate each model’s ability to identify fraud without being misled by high overall accuracy.

By comparing model performance across metrics and visualizations, this project demonstrates how to approach real-world fraud detection problems with careful evaluation strategies and interpretable, reproducible code.

**Libraries used**: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `xgboost`

---

## 📊 1. Data Exploration
- We use the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download).
- The dataset contains anonymized features (`V1–V28`), as well as `Time`, `Amount`, and `Class`.
- Only 0.17% of the transactions are labeled as fraud — making this a highly imbalanced classification problem.
- No missing values are found.

---

## 🧪 2. Data Preparation
- `X_features` contains the anonymized variables, `Time`, and `Amount`.
- `Y_output` is the target: `Class`, where `0 = legitimate`, `1 = fraud`.
- We split the data using an 80/20 ratio with `random_state=42` for reproducibility.

---

## 🤖 3. Model Training and Evaluation

### 🔹 Model 1: Logistic Regression
- A simple, interpretable baseline using `LogisticRegression(max_iter=10000)`.
- Despite very high **accuracy ≈ 0.9991**, the model initially suffers from low recall due to class imbalance.
- Key metrics:
  - **Precision (fraud)** ≈ 0.86
  - **Recall (fraud)** ≈ 0.57
  - **AUPRC**: 0.7540
- Threshold tuning (set to 0.3) improves:
  - **Recall** ↑ to 0.62
  - **Precision** ↓ slightly to 0.81
- After threshold adjustment, AUPRC improves to **0.8742**.
- Overall, logistic regression is fast and easy to interpret, but not ideal without threshold tuning.

---

### 🔹 Model 2: Random Forest
- Implemented using `RandomForestClassifier(n_estimators=25)`.
- Provides significantly better results than logistic regression:
  - **Accuracy**: 0.99958
  - **Precision (fraud)**: 0.97
  - **Recall (fraud)**: 0.78
  - **AUPRC**: 0.8613
- Random Forest detects 3 more fraud cases with only 2 additional false positives.
- It offers more robust precision across a wide range of recall values, with a flatter precision-recall curve.

---

### 🔹 Model 3: XGBoost
- A powerful gradient boosting model known for real-world performance.
- Used `XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')`.
- Metrics:
  - **Accuracy**: 0.99959
  - **Precision (fraud)**: 0.96
  - **Recall (fraud)**: 0.80
  - **AUPRC**: 0.8742
- XGBoost demonstrates the best balance between precision and recall, with the flattest and highest curve among all models.
- Ideal for production use in imbalanced problems like fraud detection.

---

## 📈 4. Model Comparison – Precision-Recall Analysis

| Model             | AUPRC   | Precision (fraud) | Recall (fraud) |
|------------------|---------|-------------------|----------------|
| Logistic (raw)   | 0.7540  | 0.86              | 0.57           |
| Logistic (thr=0.3)| 0.8742 | 0.81              | 0.62           |
| Random Forest     | 0.8613 | 0.97              | 0.78           |
| XGBoost           | 0.8742 | 0.96              | 0.80           |

- **Logistic Regression**: Easy to implement, performs poorly without threshold adjustment.
- **Random Forest**: Great balance, better recall than Logistic, slightly lower AUPRC than XGBoost.
- **XGBoost**: Best overall trade-off, flatter precision-recall curve, and highest robustness.

---

## ✅ Conclusion
XGBoost outperforms other models in terms of both AUPRC and classification metrics. However, Random Forest remains a strong contender, and Logistic Regression (with threshold tuning) is still useful as a baseline. In real-world fraud detection, boosting methods like XGBoost are often preferred due to their balance between control, precision, and generalization.
