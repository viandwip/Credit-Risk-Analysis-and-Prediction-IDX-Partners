# Credit Risk Analysis and Prediction by Using Machine Learning

## 1. Goals & Objectives
**Goals:**
1. Reducing the percentage of bad loans to below 2.5% (average Indonesia non-performing loans percentage).
2. Find out the factors that can predict whether a loan is good or bad.

**Objectives:**
1. Analyze historical data on good and bad loans to discover insights and patterns.
2. Create a machine learning classification model to predict whether a loan is good or bad.

## 2. Exploratory Data Analysis
<p align="center"><img src="images/Percentage of Loan Approved.png" alt="Percentage of Loan Approved" width = 40%></p>

### 2.1. Univariate Analysis
<p align="center"><img src="images/Univariate%20Analysis.png" alt="Univariate Analysis"></p>

#### Observation:
- The **longer** the term, the **higher** the probability of bad credit.
- **Grade A** has the **lowest** probability of bad credit and **Grade G** has the **highest** probability.
- Each **emp_length** has a fairly similar bad credit ratio with the **lowest** being **10+ years** and the **highest** being **< 1 year**.
- **MORTGAGE** home_ownership has a **lower** probability of bad credit than **OWN** and **RENT**.
- Income with **Verified** status actually has the **highest** bad credit ratio.
- The **lowest** probability of bad credit is when the loan is used for a **credit card** and the **highest** is for **small businesses**.

### 2.2. Bivariate Analysis
<p align="center"><img src="images/Bivariate%20Analysis.png" alt="Bivariate Analysis"></p>

#### Observation:
- The **longer** the term, the **higher** funded amount.
- **Grade B** has the **lowest** funded amount and **Grade G** has the **highest**.
- The **longer** emp_length, the **higher** funded amount.
- The **highest** funded amount is when home_ownership is **MORTGAGE** instead of **OWN** or **RENT**.
- Income with **Verified** status has the **highest** funded amount and **Not Verified** status has the **lowest**.
- The **highest** funded amount is when the loan is used for a **small business** and the **lowest** is for **vacation**.

## 3. Data Preprocessing
- Impute the null values with **< 1 year** for the **emp_length** column because we assumed that they don't have any employment experience, **mode** for the categorical columns, **median** for the columns that have skewed distribution, **mean** for the column that has symmetric distribution, and **Remove** the columns that have too many missing values.
- Dataset **does not have** duplicated data.
- Feature engineering for the **date related** features.
- Encode the features with **Label Encoding** if they have 2 unique values or ordinal data and with **OHE** if they have nominal data.
- Feature selection with **Mutual Information** and **Pearson Correlation**.
- Split the data into 70:30 proportions, **70% for training** and **30% for testing**.
- Conduct **standardization** process for the features used in the training and testing data.

## 4. Modeling
We will choose **precision** as our main metric because we want to minimize the **false positive**, namely people who were predicted to be able to repay the loans but apparently cannot. This is because the losses from **giving loans** to people who are **unable to repay** the loans are much greater than **not giving loans** to people who are **able to pay** the loans.

### 4.1. Model Training & Validation
| No | Model | Acc (Train) | Acc (Test) | Prec (Train) | Prec (Test) | Recall (Train) | Recall (Test) | ROC AUC (Train) | ROC AUC (Test) | Time Elapsed
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
| 1 | Decision Tree | 0.99999 | 0.98517 | 1.00000 | 0.99139	| 0.99999 | 0.99190 | 0.99999 | 0.96201 | 6.576643 |
| 2 | Random Forest |	0.99999 | 0.98384 | 0.99999 | 0.98272	| 1.00000 | 0.99936 | 0.99995 | 0.93044 | 133.361791 |
| 3 | Gradient Boosting | 0.97977 | 0.97889 | 0.97808 | 0.97718	| 0.99963 | 0.99955 | 0.91066 | 0.90781	| 124.221854 |
| 4 | Extra Trees | 0.99999 | 0.97783 | 1.00000 | 0.97627	| 0.99999 | 0.99931	| 0.99999 | 0.90394 | 82.568233 |
| 5 | Logistic Regression	| 0.97430 | 0.97360 | 0.97372 | 0.97315	| 0.99800 | 0.99778 | 0.89184 | 0.89041 | 2.956187 |
| 6 | Ada Boost	| 0.97089 | 0.97084 | 0.96915 | 0.96906 | 0.99904 | 0.99903 | 0.87297 | 0.87383 | 37.106158 |

#### Observation:
From the results above, it can be seen that **Decision Tree** is the **best model** because it has the highest Prec (Test) and **the worst** is the **Ada Boost** model because it has the lowest Prec (Test) compared to other models.

### 4.2. Hyperparameter Tuning
| No | Model | Acc (Train) | Acc (Test) | Prec (Train) | Prec (Test) | Recall (Train) | Recall (Test) | ROC AUC (Train) | ROC AUC (Test) | Time Elapsed
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
| 1 | Gradient Boosting | 0.99226 | 0.98901 | 0.99150 | 0.98831	| 0.99986 | 0.99944 | 0.96583 | 0.95312 | 1764.521223 |
| 2 | Random Forest |	0.99988 | 0.98419 | 0.99987 | 0.98301	 | 1.00000 | 0.99946 | 0.99948 | 0.93163 | 644.425480 |
| 3 | Extra Trees | 0.99604	 | 0.97599 | 0.99557 | 0.97413	 | 1.00000 | 0.99949 | 0.98228 | 0.89514	| 277.672801 |
| 4 | Ada Boost | 0.97570 | 0.97542 | 0.97401 | 0.97365	 | 0.99932 | 0.99934 | 0.89354 | 0.89310 | 1691.129956 |
| 5 | Logistic Regression	| 0.97375 | 0.97306 | 0.97308 | 0.97252	| 0.99807 | 0.99783 | 0.88916 | 0.88781 | 64.776840 |
| 6 | Decision Tree	| 0.97790 | 0.95494 | 0.98354 | 0.96942 | 0.99173 | 0.98014 | 0.92982 | 0.86823 | 11.585753 |

#### Observation:
After hyperparameter tuning there are **slightly changes** on model performances, it can be seen that **Gradient Boosting** now is the **best model** because it has the highest Prec (Test) and the **Decision Tree** model actually become the model with **the worst** performance because it has the lowest Prec (Test) compared to other models.

### 4.3. Feature Importances
<p align="center"><img src="images/Feature%20Importances.png" alt="Feature Importances" width=70%></p>

#### Observation:
Based on **feature importances** from Gradient Boosting model, the top 10 features that have the **highest contributions** in making accurate predictions are the **recoveries**, **total_rec_prncp**, **loan_duration**, **out_prncp**, **credit_report_age**, **total_rec_int**, **installment**, **total_rec_late_fee**, **grade**, and **term** features.

### 4.4. SHAP Values
<p align="center"><img src="images/SHAP%20Values.png" alt="SHAP Values" width=70%></p>

#### Observation:
Then, from the **SHAP values** we can see the impact of each feature on the model output. The features that have the higher value tend to be **good credit** namely **total_rec_prncp**, **loan_duration**, **term**, and **grade**. Meanwhile, the features that have the higher value tend to be **bad credit** namely **credit_report_age**, **installment**, **recoveries**, **out_prncp**, **total_rec_int**, and **total_rec_late_fee**.

### 4.5. Confusion Matrix
<p align="center"><img src="images/Confusion%20Matrix.png" alt="Confusion Matrix" width = 70%></p>

By using the results of *hyperparameter tuning* for the Gradient Boosting model, we train the model again to get a **confusion matrix** as shown above, with the following results:

- **True Positive**: Predicted the loan was approved and it turned out to be correct 124,057 times.
- **True Negative**: Predicted the loan was not approved and it turned out to be correct 14,289 times.
- **False Positive**: Predicted the loan was approved and turned out to be wrong by 1,461 times.
- **False Negative**: Predicted the loan was not approved and turned out to be wrong 79 times.

## 5. Business Simulation
**Before Using Machine Learning Model:**
- Good Loans = 0.888 * 466,285 = 414,061
- Bad Loans = 0.112 * 466,285 = 52,224

**After Using Machine Learning Model:**
- Good Loans = 0.988 * 466,285 = 460,690
- Bad Loans = 0.012 * 466,285 = 5,595

**Percentage:**
- Good Loans = ((460,690 - 414,061) / 414,061) * 100% = +11.26%
- Bad Loans = ((5,595 - 52,224) / 52,224) * 100% = -89.29%
****

#### Conclusion:
After using machine learning, the number of **good loans increased by 11.26%** to 98.8% or the number of **bad loans decreased by 89.29%** to 1.2%.

