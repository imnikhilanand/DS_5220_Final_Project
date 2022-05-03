# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:30:46 2022

@author: Nikhil
"""

""" importing the libraries """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# extra added for now
del project_resources_outcome = pd.read_csv("../../data/interim/project_resources_outcome.csv")
dummy = project_resources_outcome.head(100)

# reading the dataset
data_n = pd.read_csv("../../data/processed/processed_data_with_relevant_features_and_features_without_encoding_with_filtering.csv")

del data_n["index"]

# extracting the numerical features
data_n_numerical = data_n[['items_total_price', 
                           'total_items',
                           'fulfillment_labor_materials',
                         ]]

# extracting the categorical features
data_n_categorical = data_n[[
                           'school_metro_encoded', 
                           'school_charter_encoded',
                           'school_year_round_encoded', 
                           'school_nlns_encoded',
                           'school_kipp_encoded', 
                           'school_charter_ready_promise_encoded',
                           'teacher_prefix_encoded', 
                           'teacher_teach_for_america_encoded',
                           'teacher_ny_teaching_fellow_encoded', 
                           'primary_focus_subject_encoded',
                           'primary_focus_area_encoded',
                           'resource_type_encoded',
                           'poverty_level_encoded',
                           'grade_level_encoded',
                           'eligible_double_your_impact_match_encoded',
                           'eligible_almost_home_match_encoded',
                           'project_resource_type_encoded',
                           'school_state_encoded'
                         ]]

# extracting the text without stopwords
data_n_textual = data_n[["combined_text_without_stopwords"]]

# extracting the encoded class labels
data_y = data_n["is_exciting_x_encoded"]


""" build model for numerical data """

data_n_numerical.columns

# numerical data after scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_n_numerical)
data_n_numerical_scaled = scaler.transform(data_n_numerical)


from sklearn.naive_bayes import GaussianNB

X_train_numerical, X_test_numerical, y_train_numerical, y_test_numerical = train_test_split(data_n_numerical, data_y, test_size=0.1, random_state=42)

# Guassian Naive Bayes
gnb = GaussianNB(var_smoothing=1e-15)
gnb.fit(X_train_numerical, y_train_numerical)
y_pred_numerical = gnb.predict(X_test_numerical)
y_gnb_proba = gnb.predict_proba(X_test_numerical)
y_gnb_proba_df = pd.DataFrame(y_gnb_proba, columns=["num_0","num_1"]) 

confusion_matrix(y_test_numerical, y_pred_numerical)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_numerical, y_pred_numerical, target_names=target_names))

# Random Forest
random_forest_model1 = RandomForestClassifier(max_depth=1000, n_estimators= 100, random_state=0)
random_forest_model1.fit(X_train_numerical, y_train_numerical)
y_random_forest_predicted1 = random_forest_model1.predict(X_test_numerical)

confusion_matrix(y_test_numerical, y_random_forest_predicted1)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_numerical, y_random_forest_predicted1, target_names=target_names))

# Logistic Regression
logistic_regression_model = LogisticRegression(max_iter=1000, verbose=0, tol=0.0001, C=1.0)
logistic_regression_model.fit(X_train_numerical, y_train_numerical)
y_Logistic_regression_predicted = logistic_regression_model.predict(X_test_numerical)

confusion_matrix(y_test_numerical, y_Logistic_regression_predicted)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_numerical, y_Logistic_regression_predicted, target_names=target_names))

# smote to resample the data and apply the above algorithms
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_numerical, y_train_numerical)

# Guassian Naive Bayes with smote
gnb_with_smote = GaussianNB(var_smoothing=1e-15)
gnb_with_smote.fit(X_res, y_res)
y_pred_numerical = gnb_with_smote.predict(X_test_numerical)
y_gnb_proba = gnb_with_smote.predict_proba(X_test_numerical)
y_gnb_proba_df = pd.DataFrame(y_gnb_proba, columns=["num_0","num_1"]) 

confusion_matrix(y_test_numerical, y_pred_numerical)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_numerical, y_pred_numerical, target_names=target_names))

# random forest with smote - POOR PERFORMANCE
random_forest_model1_smote = RandomForestClassifier(max_depth=1000, n_estimators= 100, random_state=0)
random_forest_model1_smote.fit(X_res, y_res)
y_random_forest_predicted1 = random_forest_model1_smote.predict(X_test_numerical)

confusion_matrix(y_test_numerical, y_random_forest_predicted1)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_numerical, y_random_forest_predicted1, target_names=target_names))

# Logistic Regression with smote applied
logistic_regression_model_smote = LogisticRegression(max_iter=1000, verbose=0,tol=0.0001, C=1.0)
logistic_regression_model_smote.fit(X_res, y_res)
y_Logistic_regression_predicted = logistic_regression_model_smote.predict(X_test_numerical)
y_gnb_proba = logistic_regression_model_smote.predict_proba(X_test_numerical)
y_gnb_proba_df = pd.DataFrame(y_gnb_proba, columns=["num_0","num_1"]) 


confusion_matrix(y_test_numerical, y_Logistic_regression_predicted)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_numerical, y_Logistic_regression_predicted, target_names=target_names))




""" model for categorical data """

X_train_categorical, X_test_categorical, y_train_categorical, y_test_categorical = train_test_split(data_n_categorical, data_y, test_size=0.1, random_state=42)

from sklearn.naive_bayes import ComplementNB

#complementNB
nb_model_cat = ComplementNB(alpha=1, norm=False)
nb_model_cat.fit(X_train_categorical, y_train_categorical)
y_nb_predicted = nb_model_cat.predict(X_test_categorical)
y_nb_categorical_proba = nb_model_cat.predict_proba(X_test_categorical)
y_nb_cat_proba_df = pd.DataFrame(y_nb_categorical_proba, columns=["cat_0","cat_1"]) 

confusion_matrix(y_test_categorical, y_nb_predicted)
target_names = ['class 0', 'class 1']
print(classification_report(y_nb_predicted, y_test_categorical, target_names=target_names))

# multinomialNB
m_nb_model_cat = MultinomialNB()
m_nb_model_cat.fit(X_train_categorical, y_train_categorical)
y_nb_predicted = m_nb_model_cat.predict(X_test_categorical)
y_nb_categorical_proba = m_nb_model_cat.predict_proba(X_test_categorical)
y_nb_cat_proba_df = pd.DataFrame(y_nb_categorical_proba, columns=["cat_0","cat_1"]) 

confusion_matrix(y_test_categorical, y_nb_predicted)
target_names = ['class 0', 'class 1']
print(classification_report(y_nb_predicted, y_test_categorical, target_names=target_names))

# random forest
random_model_cat = RandomForestClassifier(max_depth=100, n_estimators= 100, random_state=0)
random_model_cat.fit(X_train_categorical, y_train_categorical)
random_predicted = random_model_cat.predict(X_test_categorical)
y_nb_categorical_proba = random_model_cat.predict_proba(X_test_categorical)
y_nb_cat_proba_df = pd.DataFrame(y_nb_categorical_proba, columns=["cat_0","cat_1"]) 

confusion_matrix(y_test_categorical, random_predicted)
target_names = ['class 0', 'class 1']
print(classification_report(random_predicted, y_test_categorical, target_names=target_names))



""" model for text data """

from sklearn.feature_extraction.text import TfidfVectorizer

text = data_n_textual["combined_text_without_stopwords"].values
vectorizer = TfidfVectorizer(max_features=4500)
X = vectorizer.fit_transform(text)
X = X.toarray()

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X, data_y, test_size=0.1, random_state=42)

from sklearn.naive_bayes import ComplementNB

# complement naive bayes
nb_model_text = ComplementNB(alpha=0.1, norm=False)
nb_model_text.fit(X_train_text, y_train_text)
y_nb_text_predicted = nb_model_text.predict(X_test_text)
y_nb_text_proba = nb_model_text.predict_proba(X_test_text)
y_nb_text_proba_df = pd.DataFrame(y_nb_text_proba, columns=["text_0","text_1"]) 

confusion_matrix(y_test_text, y_nb_text_predicted)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_text, y_nb_text_predicted, target_names=target_names))

# multinomial
m_nb_model_text = MultinomialNB(alpha=0.1)
m_nb_model_text.fit(X_train_text, y_train_text)
y_nb_text_predicted = m_nb_model_text.predict(X_test_text)
y_nb_text_proba = m_nb_model_text.predict_proba(X_test_text)
y_nb_text_proba_df = pd.DataFrame(y_nb_text_proba, columns=["text_0","text_1"]) 

confusion_matrix(y_test_text, y_nb_text_predicted)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_text, y_nb_text_predicted, target_names=target_names))



""" Stacking the three results"""

y_gnb_proba_df = y_gnb_proba_df.join(y_nb_cat_proba_df)
y_gnb_proba_df = y_gnb_proba_df.join(y_nb_text_proba_df)

# modeling the data
X_train_com, X_test_com, y_train_com, y_test_com = train_test_split(y_gnb_proba_df, y_test_text, test_size=0.1, random_state=42, stratify=y_test_text)

# Gaussian Naive Bayes
gnb2 = GaussianNB()
gnb2.fit(X_train_com, y_train_com)
y_pred_comb = gnb2.predict(X_test_com)

confusion_matrix(y_test_com, y_pred_comb)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_com, y_pred_comb, target_names=target_names))

# Random Forest
random_forest_final = RandomForestClassifier(max_depth=1000, n_estimators= 100, random_state=0)
random_forest_final.fit(X_train_com, y_train_com)
y_pred_comb = random_forest_final.predict(X_test_com)

confusion_matrix(y_test_com, y_pred_comb)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_com, y_pred_comb, target_names=target_names))

# Logisitc Regression
lr_without_smote = LogisticRegression(max_iter=1000, verbose=0,tol=0.0001, C=1.0)
lr_without_smote.fit(X_train_com, y_train_com)
y_pred_comb = lr_without_smote.predict(X_test_com)

confusion_matrix(y_test_com, y_pred_comb)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_com, y_pred_comb, target_names=target_names))


# implementing SMOTE
from imblearn.over_sampling import SMOTE
sm2 = SMOTE(random_state=42)
X_res2, y_res2 = sm2.fit_resample(X_train_com, y_train_com)

# Gaussian Naive Bayes with smote
gnb3 = GaussianNB()
gnb3.fit(X_res2, y_res2)
y_pred_comb = gnb3.predict(X_test_com)

confusion_matrix(y_test_com, y_pred_comb)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_com, y_pred_comb, target_names=target_names))

# Random Forest with SMOTE
random_forest_final_smote = RandomForestClassifier(max_depth=1000, n_estimators= 100, random_state=0)
random_forest_final_smote.fit(X_res2, y_res2)
y_pred_comb = random_forest_final.predict(X_test_com)

confusion_matrix(y_test_com, y_pred_comb)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_com, y_pred_comb, target_names=target_names))

# Logisitc Regression with SMOTE
lr_smote = LogisticRegression(max_iter=100, verbose=0,tol=0.1, C=1.0)
lr_smote.fit(X_res2, y_res2)
y_pred_comb = lr_smote.predict(X_test_com)

confusion_matrix(y_test_com, y_pred_comb)
target_names = ['class 0', 'class 1']
print(classification_report(y_test_com, y_pred_comb, target_names=target_names))







# extras
predictions_probabability = gnb3.predict_proba(X_test_com)


predictions_probabability_df = pd.DataFrame(predictions_probabability, columns=["0_prob","1_prob"])

predictions_probabability_df["actual"] = y_pred_comb

data_roc = predictions_probabability_df.query("actual == 1")


# ROC curve
metrics.plot_roc_curve(gnb3, X_test_com, y_test_com)


from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(data_roc["actual"], data_roc["1_prob"], pos_label=2)

thresolds



