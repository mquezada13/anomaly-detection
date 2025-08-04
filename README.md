# Project: Credit Card Fraud Detection

**Author**: Maura E. Ramirez-Quezada  
**Description**: This notebook presents a study of credit card transaction data, focusing on the detection of fraudulent activity.  We train models using different approaches, starting with a simple and interpretable method for binary classification.

**Libraries used**: pandas, numpy, scikit-learn,seaborn, matplotlib(to be extended)

## Structure:
### Data Exploration 
- We import the dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
- We explore the dataset structure, including dimensions, number of columns, total transactions, and missing values.
- We analyze the class distribution and identify that only 0.17% of the transactions are labeled as fraudulent.
### Sorting data for training
- We define the features `X_features` and target `Y_output`.
- `X_features` includes the anonymized columns V1–V28, plus `Time` and `Class`.
- `Y_output` corresponds to the `Class` column, where 0 indicates a legitimate transaction and 1 indicates fraud.
- We split the dataset into training and test sets: (`X_train`, `y_train`) and (`X_test`, `y_test`) to evaluate model performance on unseen data.
- The data is split with an 80/20 ratio, using `random_state=42` for reproducibility.
### Training
- Model 1: Logistic Regresion. 
    - This is using ` LogisticRegression()` from scikit-learn as our baseline model.
    - We increase “max_iter= 10000` to ensure conversion.
    - The model is evaluated using `accuracy_score`, `confusion_matrix`, `classification_report`
    - Analyse the output: The model works but with low accurancy.
        - Accuracy ≈ 0.9991 (misleading due to class imbalance)
        - Precision (fraud class) ≈ 0.86
        - Recall (fraud class) ≈ 0.57
    - We then compute predicted probabilities for the positive class (fraud).
    - Using these probabilities, we generate the Precision-Recall curve and calculate the Area Under the Curve: **AUPRC = 0.7540**. On average, the model maintains 75% precision as recall increases.
    - To improve fraud detection, we manually lower the decision threshold to 0.3 and re-evaluate the model.
        - This adjustment increases recall from 0.57 to 0.62, while slightly reducing precision from 0.86 to 0.81.
        - A reasonable trade-off when detecting fraud is more critical than avoiding false positives.

    - Although the model captures many fraudulent transactions, others are still missed. We will explore additional models to improve performance, particularly recall.
- Model 2: Random Forest
    - This model uses `RandomForestClassifier` from scikit-learn with 25 trees (`n_estimators=25`). 30 & 50 were checked but gave not better results.
    - The accuracy of this model is significantly better than logistic regression: **0.99958**.
    - It detects **76 fraudulent transactions (TP)** and misses **22 (FN)**.
    - **Precision**: 0.97 | **Recall**: 0.78
    - Compared to logistic regression, this model detects **3 more fraud cases** with only **2 false positives**.
    - The **AUPRC = 0.8613**, which is excellent for such a highly imbalanced classification problem.
    - The curve is more stable and flatter than in the logistic regression case, indicating more consistent confidence across thresholds.
    - Random Forest demonstrates higher performance and confidence when detecting fraud cases.


_(To be continued...)_
