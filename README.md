# Project: Credit Card Fraud Detection

**Author**: Maura E. Ramirez-Quezada  
**Description**: This notebook presents a study of credit card transaction data, focusing on the detection of fraudulent activity.  
**Libraries used**: pandas, numpy, scikit-learn,  (to be extended)

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

_(To be continued...)_
