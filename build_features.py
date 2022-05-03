# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 09:43:58 2022

@author: Nikhil
"""

""" IMPORTING THE LIBRARIES """
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords

""" LOADING THE DATA """
# project_resources_outcome
project_resources_outcome = pd.read_csv("../../data/interim/project_resources_outcome.csv")
# essay file 
essay = pd.read_csv("../../data/raw/essays/essays.csv")


""" PREPROCESSING """

# removing primary keys (irrelevant for modeling)
del project_resources_outcome["vendorid"]
del project_resources_outcome["vendor_name"]
del project_resources_outcome["item_name"]
del project_resources_outcome["item_number"]
del project_resources_outcome["resourceid"]
del project_resources_outcome["school_district"]

# grouping the data
temp = project_resources_outcome.groupby(['projectid']).aggregate(items_total_price=('item_unit_price','sum'), total_items=('item_quantity','sum'))
# renaming the axis
temp = temp.rename_axis('projectid').reset_index()
# merging the tables
temp = temp.merge(project_resources_outcome, left_on="projectid", right_on="projectid", how='inner')
# deleteing the irrelavant columns
del temp["item_unit_price"]
del temp["item_quantity"]
# dropping the duplicates
temp = temp.drop_duplicates()
# deleteing table
del project_resources_outcome

""" BUILDING FEATURES """

# checking the nan values in each column
for i in temp.columns:
    temp2 = temp[i].isna().sum()
    if temp2 > 0:
        print(i," -> ", temp2, " -> ", temp2/619001)


""" FEATURE SELECTION """

# only those data points where there is no duplicate values
temp2 = temp.dropna(how='any')

# encoding features
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# function to label encode features
def label_encode_features(data, x):
    le = LabelEncoder()
    le_encoded = le.fit_transform(data[x])
    le_encoded = le_encoded.reshape(len(le_encoded), 1)
    data[x+"_encoded"] = le_encoded
    return data

# categorical features
categorical_features = ['school_metro', 
                        'school_county', 
                        'school_charter',
                        'school_magnet', 
                        'school_year_round', 
                        'school_nlns', 
                        'school_kipp',
                        'school_charter_ready_promise', 
                        'teacher_prefix',
                        'teacher_teach_for_america', 
                        'teacher_ny_teaching_fellow',
                        'primary_focus_subject', 
                        'primary_focus_area',
                        'secondary_focus_subject', 
                        'secondary_focus_area', 
                        'resource_type',
                        'poverty_level', 
                        'grade_level', 
                        'eligible_double_your_impact_match', 
                        'eligible_almost_home_match',
                        'project_resource_type', 
                        'is_exciting']

# label encoding different features
for i in categorical_features:
    temp2 = label_encode_features(temp2, i)
    
# CHI_SQUARED TEST
def check_p_Chi_square(a,b,data):
    temp = pd.crosstab(data[a],data[b],margins=False)
    stat, p, dof, expected = chi2_contingency(temp)
    return p

for i in categorical_features:
    if i != 'is_exciting_encoded':
        score = check_p_Chi_square(i, 'is_exciting_encoded', temp2)
        print(i," -> ",score)

###############################################################################
# At the threshold of 0.01, all the categorical features are significant 

# school_metro 
# school_county 
# school_charter
# school_year_round 
# school_nlns  
# school_kipp
# school_charter_ready_promise 
# teacher_prefix
# teacher_teach_for_america 
# teacher_ny_teaching_fellow
# primary_focus_subject 
# primary_focus_area
# secondary_focus_subject 
# secondary_focus_area', 
# resource_type
# poverty_level
# grade_level 
# eligible_double_your_impact_match
# eligible_almost_home_match
# project_resource_type

###############################################################################

# ANOVA TEST
def check_p(a,b,data):
    temp = data[[a,b]]
    grps = pd.unique(temp[b].values)
    d_data = {grp:temp[a][temp[b] == grp] for grp in grps} 
    F, p = f_oneway(d_data[0], d_data[1])
    return p

numerical_features = ['items_total_price', 
                      'total_items',
                      'fulfillment_labor_materials',
                      'total_price_excluding_optional_support',
                      'total_price_including_optional_support', 
                      'students_reached'
                      ]

for i in numerical_features:
    p = check_p(i, 'is_exciting_encoded', temp2)
    print(i, " -> ", p)

###############################################################################
# At the threshold of 0.01, the numerical features that are significant -
# items_total_price
# total_items
# fulfillment_labor_materials
###############################################################################


""" DATA PREPROCESSING """

# checking the unique values in each category

for i in categorical_features:
    print(i," -> ",len(temp2[i].unique()))


###############################################################################
# Since the number of unique counties are mode than 1500, we cannot encode it
# Secondary focus subject and are not there in 32% cases we are dropping them as well
###############################################################################

# selecting relevant columns after statistical test and possibility of missing value imputation

temp = temp[['projectid', 
            'items_total_price', 
            'total_items', 
            'school_metro', 
            'school_charter', 
            'school_year_round', 
            'school_nlns',
            'school_kipp', 
            'school_charter_ready_promise', 
            'teacher_prefix',
            'teacher_teach_for_america', 
            'teacher_ny_teaching_fellow',
            'primary_focus_subject', 
            'primary_focus_area',
            'resource_type',
            'poverty_level', 
            'grade_level', 
            'fulfillment_labor_materials',
            'eligible_double_your_impact_match', 
            'eligible_almost_home_match',
            'project_resource_type', 
            'is_exciting']]


# Adding the missing state columns

# !!!! IMPORTANT

# uploading the projects to find the states column

projects = pd.read_csv("../../data/raw/projects/projects.csv")

projects = projects[["projectid","school_state"]]

temp = temp.merge(projects, left_on="projectid", right_on="projectid", how="left")

score = check_p_Chi_square('school_state', 'is_exciting_encoded', temp2)

temp2 = temp[['school_state','is_exciting']]

temp2 = label_encode_features(temp2, 'school_state')
temp2 = label_encode_features(temp2, 'is_exciting')

check_p_Chi_square('is_exciting_encoded', 'school_state_encoded',temp2)

# Since the p value is 0.0, the school state is relevant

# saving the processed file without encoding
temp.to_csv("../../data/processed/processed_data_with_relevant_features_without_encoding.csv", index=False)


""" Dropping coulums """

data = pd.read_csv("../../data/processed/processed_data_with_relevant_features_without_encoding.csv")

#checking missing values again !
for i in data.columns:
    temp2 = data[i].isna().sum()
    if temp2 > 0:
        print(i," -> ", temp2, " -> ", temp2/6190)

# imputing the missing school metro
dummy = data.head(100)

projects = pd.read_csv("../../data/raw/projects/projects.csv")

# dropping projects where school_metro is null
data = data.query("school_metro.notnull()", engine="python")

# dropping projects where project_resource_type
data = data.query("project_resource_type.notnull()", engine="python")

# dropping projects where there is not grade level
data = data.query("grade_level.notnull()", engine="python")

# dropping projects where there is not primary focus subject
data = data.query("primary_focus_subject.notnull()", engine="python")

# dropping projects where there is no reference_type
data = data.query("reference_type.notnull()", engine="python")

# dropping projects where there is no fulfillment_labor_material
data = data.query("fulfillment_labor_materials.notnull()", engine="python")

# saving data after dropping duplicates
data.to_csv("../../data/processed/processed_data_with_relevant_features_without_encoding_dropped_duplicates.csv", index=False)


""" Label Encoding the columns """

# encoding school_metro
data = label_encode_features(data, 'school_metro')

# encoding school_charter
data = label_encode_features(data, 'school_charter')

# encoding school_year_around
data = label_encode_features(data, 'school_year_round')

# encoding school_year_around
data = label_encode_features(data, 'school_charter')

# encoding school_nlns
data = label_encode_features(data, 'school_nlns')

# encoding school_kipp
data = label_encode_features(data, 'school_kipp')

# encoding school_kipp
data = label_encode_features(data, 'school_charter_ready_promise')

# encoding teacher_teach_for_america
data = label_encode_features(data, 'teacher_prefix')

# encoding teacher_prefix
data = label_encode_features(data, 'teacher_teach_for_america')

# encoding teacher_ny_teaching_fellow
data = label_encode_features(data, 'teacher_ny_teaching_fellow')

# encoding primary_focus_subject
data = label_encode_features(data, 'primary_focus_subject')

# encoding resource type
data = label_encode_features(data, 'resource_type')

# encoding poverty_level
data = label_encode_features(data, 'poverty_level')

# encoding grade_level
data = label_encode_features(data, 'grade_level')

# encoding eligible_doubkle_your_impact_match
data = label_encode_features(data, 'eligible_double_your_impact_match')

# encoding eligible_almost_home_match
data = label_encode_features(data, 'eligible_almost_home_match')

# encoding project_resource_type
data = label_encode_features(data, 'project_resource_type')

# encoding is_exiciting
data = label_encode_features(data, 'is_exciting')

# encoding is_exiciting
data = label_encode_features(data, 'primary_focus_area')

# encoding school_state
data = label_encode_features(data, 'school_state')

# fetching only numerical and encoded columns
data2 = data[['projectid', 
              'school_metro_encoded',
              'items_total_price', 
              'total_items',
              'fulfillment_labor_materials',
              'school_metro_encoded', 'school_charter_encoded',
              'school_year_round_encoded', 'school_nlns_encoded',
              'school_kipp_encoded', 'school_charter_ready_promise_encoded',
              'teacher_prefix_encoded', 'teacher_teach_for_america_encoded',
              'teacher_ny_teaching_fellow_encoded', 'primary_focus_subject_encoded', 'primary_focus_area_encoded',
              'resource_type_encoded', 'poverty_level_encoded', 'grade_level_encoded',
              'eligible_double_your_impact_match_encoded',
              'eligible_almost_home_match_encoded', 'project_resource_type_encoded','school_state_encoded',
              'is_exciting_encoded']]


# saving data after one-hot encoding
data2.to_csv("../../data/processed/processed_data_with_relevant_features_encoding_dropped_duplicates.csv", index=False)

data_numerical = pd.read_csv("../../data/processed/processed_data_with_relevant_features_encoding_dropped_duplicates.csv")

""" BUILDING TEXT FEATURES """

# loading the dataset
data_text = pd.read_csv("../../data/raw/essays/essays.csv")

# selecting only relevant rows and columns for trainingg and test set
project_ids = data_numerical[["projectid"]]
data_text = project_ids.merge(data_text, left_on='projectid', right_on='projectid', how="inner")

# merging the text rows
def concatenate_text(x):
    return str(x[2])+" "+ str(x[3])+" "+ str(x[4])+" "+ str(x[5])

data_text["combined_text"] = data_text.apply(concatenate_text, axis=1) 

# lowering the string
def lower_string(x):
    return str(x).lower()

data_text["combined_text_lower"] = data_text["combined_text"].apply(lambda x: lower_string(x)) 

# remove \na and \r
def remove_n(x):
    x = re.sub("\\\\n", " ", str(x))
    x = re.sub("\\n", " ", str(x))
    x = re.sub("\\\r", " ", str(x))
    x = re.sub("\\r"," ",str(x))
    return x

data_text["combined_text_without_n"] = data_text["combined_text_lower"].apply(lambda x: remove_n(x)) 

# handle contractions
contractions = {
                "ain't": "am not",
                "aren't": "are not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "couldn't've": "could not have",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hadn't've": "had not have",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he would",
                "he'd've": "he would have",
                "he'll": "he will",
                "he'll've": "he will have",
                "he's": "he is",
                "how'd": "how did",
                "how'd'y": "how do you",
                "how'll": "how will",
                "how's": "how is",
                "i'd": "i would",
                "i'd've": "i would have",
                "i'll": "i will",
                "i'll've": "i will have",
                "i'm": "i am",
                "i've": "i have",
                "isn't": "is not",
                "it'd": "it would",
                "it'd've": "it would have",
                "it'll": "it will",
                "it'll've": "it will have",
                "it's": "it is",
                "let's": "let us",
                "ma'am": "madam",
                "mayn't": "may not",
                "might've": "might have",
                "mightn't": "might not",
                "mightn't've": "might not have",
                "must've": "must have",
                "mustn't": "must not",
                "mustn't've": "must not have",
                "needn't": "need not",
                "needn't've": "need not have",
                "o'clock": "of the clock",
                "oughtn't": "ought not",
                "oughtn't've": "ought not have",
                "shan't": "shall not",
                "sha'n't": "shall not",
                "shan't've": "shall not have",
                "she'd": "she would",
                "she'd've": "she would have",
                "she'll": "she will",
                "she'll've": "she will have",
                "she's": "she is",
                "should've": "should have",
                "shouldn't": "should not",
                "shouldn't've": "should not have",
                "so've": "so have",
                "so's": "so is",
                "that'd": "that would",
                "that'd've": "that would have",
                "that's": "that is",
                "there'd": "there would",
                "there'd've": "there would have",
                "there's": "there is",
                "they'd": "they would",
                "they'd've": "they would have",
                "they'll": "they will",
                "they'll've": "they will have",
                "they're": "they are",
                "they've": "they have",
                "to've": "to have",
                "wasn't": "was not",
                "we'd": "we would",
                "we'd've": "we would have",
                "we'll": "we will",
                "we'll've": "we will have",
                "we're": "we are",
                "we've": "we have",
                "weren't": "were not",
                "what'll": "what will",
                "what'll've": "what will have",
                "what're": "what are",
                "what's": "what is",
                "what've": "what have",
                "when's": "when is",
                "when've": "when have",
                "where'd": "where did",
                "where's": "where is",
                "where've": "where have",
                "who'll": "who will",
                "who'll've": "who will have",
                "who's": "who is",
                "who've": "who have",
                "why's": "why is",
                "why've": "why have",
                "will've": "will have",
                "won't": "will not",
                "won't've": "will not have",
                "would've": "would have",
                "wouldn't": "would not",
                "wouldn't've": "would not have",
                "y'all": "you all",
                "y'all'd": "you all would",
                "y'all'd've": "you all would have",
                "y'all're": "you all are",
                "y'all've": "you all have",
                "you'd": "you would",
                "you'd've": "you would have",
                "you'll": "you will",
                "you'll've": "you will have",
                "you're": "you are",
                "you've": "you have"
            }

def handle_contractions(x):
    for words in contractions:
        x = re.sub(words," "+contractions[words]+" ",x)
    return x

data_text["combined_text_contractions"] = data_text["combined_text_without_n"].apply(lambda x: handle_contractions(x)) 

# remove punctuation
def remove_punctuation(x):
    x = x.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))
    return x

data_text["combined_text_without_punc"] = data_text["combined_text_contractions"].apply(lambda x: remove_punctuation(x)) 

# remove numbers
def remove_numbers(x):
    x = re.sub('[0-9]', '', x)
    return x
    
data_text["combined_text_without_num"] = data_text["combined_text_without_punc"].apply(lambda x: remove_numbers(x))    

# lemmatization

wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize_text(x):
    word_tokens = nltk.word_tokenize(x)
    temp_var_1 = []
    for words in word_tokens:
        temp_var_1.append(wordnet_lemmatizer.lemmatize(words,pos='v'))
        temp_var_1.append(" ")
    x = ''.join(temp_var_1).strip()
    return x


data_text["combined_text_lemma_new"] = data_text["combined_text_lemma"].apply(lambda x: lemmatize_text(x))    


# removing irrelvant rows
del data_text["combined_text_lower"]
del data_text["combined_text_without_n"]
del data_text["combined_text_contractions"]
del data_text["combined_text_without_punc"]
del data_text["combined_text_without_num"]


#saving the text features dataframe
                                                  
# remove_stopwords
stopwords = set(stopwords.words('english'))

def remove_stopwords(x):
    x_tokens = nltk.word_tokenize(x)
    temp = []
    for i in x_tokens:
        if i not in stopwords:
            temp.append(i)
    return ' '.join(temp).strip()

data_text["combined_text_without_stopwords"] = data_text["combined_text_lemma_new"].apply(lambda x: remove_stopwords(x))


# dropping irrelevant columns
data_text.columns

# deleting irrelevant columns
del data_text["combined_text_lemma"]
del data_text["combined_text_lemma_new"]

# saving the pdf file
data_text.to_csv("../../data/processed/processed_text_data_without_stopwords.csv", index=False)






