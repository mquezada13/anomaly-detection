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
        - Although the model captures some fraudulent transactions, many are still missed. We will explore other techniques to improve recall.

_(To be continued...)_
