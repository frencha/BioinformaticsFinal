#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:42:57 2019

@author: NCBI Hackathon members and Anne French
"""

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from logitboost import LogitBoost
import datetime
import numpy as np

def model_comp(X_train, X_test, y_train, y_test, title = ""):
    xgboost_model = XGBClassifier(learning_rate = 0.01, max_depth = 3, n_estimators = 700, random_state=8)
    gradient_boost_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=4, max_features='log2', min_samples_leaf=4, n_estimators=280, subsample=0.25, random_state=8)
    random_forest_model = RandomForestClassifier(n_estimators=300, max_depth=3, verbose=1, random_state=8)
    svm_model = SVC(kernel='poly', probability=True, verbose=1, random_state=8)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    elm_model = MLPClassifier(hidden_layer_sizes=(80, ), activation='logistic', learning_rate_init=0.01, verbose=1)
    adaboost_model = AdaBoostClassifier(n_estimators=300, learning_rate=0.01, random_state=8)
    logitboost_model = LogitBoost(n_estimators=300, learning_rate=0.01, random_state=8)
    
    xgboost_model.fit(X_train, y_train)
    gradient_boost_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    elm_model.fit(X_train, y_train)
    adaboost_model.fit(X_train, y_train)
    logitboost_model.fit(X_train, y_train)

    p_random_forest = random_forest_model.predict_proba(X_test)
    p_gradient_boost =  gradient_boost_model.predict_proba(X_test)
    p_xgboost = xgboost_model.predict_proba(X_test)
    p_svm = svm_model.predict_proba(X_test)
    p_knn = knn_model.predict_proba(X_test)
    p_elm = elm_model.predict_proba(X_test)
    p_adaboost = adaboost_model.predict_proba(X_test)
    p_logitboost = logitboost_model.predict_proba(X_test)
    
    random_forest_ll = log_loss(y_test, p_random_forest)
    gradient_boost_ll = log_loss(y_test, p_gradient_boost)
    xgboost_ll = log_loss(y_test, p_xgboost)
    svm_ll = log_loss(y_test, p_svm)
    knn_ll = log_loss(y_test, p_knn)
    elm_ll = log_loss(y_test, p_elm)
    adaboost_ll = log_loss(y_test, p_adaboost)
    logitboost_ll = log_loss(y_test, p_logitboost)
    
    strng0 = "\n"+title
    strtest = "\nLength of test data: " + str(len(y_test))
    strng2 = "\n------------------"
    strng4 = "\nGradient Boost Log Loss " + str(gradient_boost_ll)
    strng5 = "\nRandom Forest Log Loss " + str(random_forest_ll)
    strng6 = "\nXGBoost Log Loss " + str(xgboost_ll)
    strng7 = "\n------------------"
    strng9 = "\nSVM Log Loss " + str(svm_ll)
    strng10 = "\nKNN Log Loss " + str(knn_ll)
    strng11 = "\nELM Log Loss " + str(elm_ll)
    strng12 = "\nAdaBoost Log Loss " + str(adaboost_ll)
    strng13 = "\nLogitBoost Log Loss " + str(logitboost_ll)
    prntstr = strng0 + strtest + strng2 + strng4 + strng5 + strng6 + strng7 + strng9 + strng10 + strng11 + strng12 +strng13
    print(prntstr)
    write_to_file(prntstr)
    
    return xgboost_model, random_forest_model, adaboost_model


def ensemble_comp(X_train, X_test, y_train, y_test, title):
    clf1 = XGBClassifier(learning_rate = 0.01, max_depth = 3, n_estimators = 700, random_state=8)
    clf2 = GradientBoostingClassifier(learning_rate=0.01, max_depth=4, max_features='log2', min_samples_leaf=4, n_estimators=280, subsample=0.25, random_state=8)
    clf3 = RandomForestClassifier(n_estimators=300, max_depth=3, verbose=1, random_state=8)
    clf4 = SVC(kernel='poly', probability=True, verbose=1, random_state=8)
    clf5 = KNeighborsClassifier(n_neighbors=3)
    clf6 = MLPClassifier(hidden_layer_sizes=(80, ), activation='logistic', learning_rate_init=0.01, verbose=1)
    clf7 = AdaBoostClassifier(n_estimators=300, learning_rate=0.01, random_state=8)
    clf8 = LogitBoost(n_estimators=300, learning_rate=0.01, random_state=8)
    complete_voting_model = VotingClassifier(estimators=[('xgb', clf1), ('gb', clf2), ('rf', clf3), ('svm', clf4), ('knn', clf5), ('elm', clf6), ('ada', clf7), ('logit', clf8)], voting='soft')
    new_voting_model = VotingClassifier(estimators=[('svm', clf4), ('knn', clf5), ('elm', clf6), ('ada', clf7), ('logit', clf8)], voting='soft')
    new_voting_model_without_logit = VotingClassifier(estimators=[('svm', clf4), ('knn', clf5), ('elm', clf6), ('ada', clf7)], voting='soft')
    original_voting_model = VotingClassifier(estimators=[('xgb', clf1), ('gb', clf2), ('rf', clf3)], voting='soft')
    
    complete_voting_model.fit(X_train, y_train)
    new_voting_model.fit(X_train, y_train)
    new_voting_model_without_logit.fit(X_train, y_train)
    original_voting_model.fit(X_train, y_train)
    
    p_complete_voting = complete_voting_model.predict_proba(X_test)
    p_new_voting = new_voting_model.predict_proba(X_test)
    p_new_voting_without_logit = new_voting_model_without_logit.predict_proba(X_test)
    p_original_voting = original_voting_model.predict_proba(X_test)
    
    complete_voting_ll = log_loss(y_test, p_complete_voting)
    new_voting_ll = log_loss(y_test, p_new_voting)
    new_voting_without_logit_ll = log_loss(y_test, p_new_voting_without_logit)
    original_voting_ll = log_loss(y_test, p_original_voting)
    
    #reset models so that overfitting doesn't occur 
    complete_voting_model = VotingClassifier(estimators=[('xgb', clf1), ('gb', clf2), ('rf', clf3), ('svm', clf4), ('knn', clf5), ('elm', clf6), ('ada', clf7), ('logit', clf8)], voting='soft')
    new_voting_model = VotingClassifier(estimators=[('svm', clf4), ('knn', clf5), ('elm', clf6), ('ada', clf7), ('logit', clf8)], voting='soft')
    new_voting_model_without_logit = VotingClassifier(estimators=[('svm', clf4), ('knn', clf5), ('elm', clf6), ('ada', clf7)], voting='soft')
    original_voting_model = VotingClassifier(estimators=[('xgb', clf1), ('gb', clf2), ('rf', clf3)], voting='soft')

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test))
    complete_voting_accuracy = np.mean(cross_val_score(complete_voting_model, X, y, cv=5))
    new_voting_accuracy = np.mean(cross_val_score(new_voting_model, X, y, cv=5))
    new_voting_without_logit_accuracy = np.mean(cross_val_score(new_voting_model_without_logit, X, y, cv=5))
    original_voting_accuracy = np.mean(cross_val_score(original_voting_model, X, y, cv=5))
    
    titlestr = "\n"+title
    str2 = "\nTotal Ensemble Log Loss " + str(complete_voting_ll)
    str3 = "\nNew Ensemble Log Loss " + str(new_voting_ll)
    str4 = "\nNew Ensemble without LogitBoost Log Loss " + str(new_voting_without_logit_ll)
    str5 = "\nOriginal Ensemble (Gradient Boost, Random Forest, and XGBoost) Log Loss " + str(original_voting_ll)
    str6 = "\n\nTotal Ensemble Mean Cross Fold Accuracy " + str(complete_voting_accuracy)
    str7 = "\nNew Ensemble Mean Cross Fold Accuracy " + str(new_voting_accuracy)
    str8 = "\nNew Ensemble without LogitBoost Cross Mean Fold Accuracy " + str(new_voting_without_logit_accuracy)
    str9 = "\nOriginal Ensemble (Gradient Boost, Random Forest, and XGBoost) Mean Accuracy " + str(original_voting_accuracy)
    lenstr = "\nAverage size of fold: " + str(len(y)/5)
    printstr = titlestr+str2+str3+str4+str5+str6+str7+str8+str9+lenstr
    print(printstr)
    write_to_file(printstr)
    

def write_to_file(printstr):
    with open("output.txt","a") as f:
        f.write("\n\n\n" + "---------------------------------------------\n" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        f.write(printstr)
        f.write("\n")


def data_prep_columns(df, var):
    """

    Parameters
    ----------------
    df: dataframe of patients
    var: str either 'Max' or 'Min', sets whether to take top 1000 columns of
    max variance or minimum variance

    Returns
    ----------------
    df2: transformed dataframe with 'Diagostic ID' and no 'Unknown' values for risk_group
    new column 'Low Risk' with boolean value 0 or 1; THIS COLUMN HAS ALL COLUMNS (clinical data, manifest data)

    low_columns_to_take_full: columns (genes) of either most/least variance from the two subgroups
    """
    if var =='Max':
        reverse_ = True
    else:
        reverse_ = False

    df1 = df[df['Diagnostic ID'].isin(('09A', '03A', '01A'))]
    df2 = df1[df1['Risk group'] != 'Unknown'].copy()
    df2['Low Risk'] = df2['Risk group'].apply(lambda x: (x == 'Low') * 1)

    low_df = df2[df2['Low Risk'] == 1].copy()
    ldf = low_df[low_df.columns[84:-4]].copy()
    ldf_var = list(enumerate(ldf.var(axis=0)))
    highest_var_low = sorted(ldf_var, key=lambda x:x[1], reverse=reverse_)[:1000]

    low_columns_to_take = [indx for indx, val in highest_var_low]
    low_columns_to_take_full = [item + 84 for item in low_columns_to_take]

    high_df = df2[df2['Low Risk'] == 0].copy()
    hdf = high_df[high_df.columns[84:-4]].copy()

    hdf_var = list(enumerate(hdf.var(axis=0)))
    highest_var_high = sorted(hdf_var, key=lambda x:x[1], reverse=reverse_)[:1000]
    
    high_columns_to_take = [indx for indx, val in highest_var_high]
    high_columns_to_take_full = [item + 84 for item in high_columns_to_take]
    low_columns_to_take_full.extend(x for x in high_columns_to_take_full if x not in low_columns_to_take_full)
    return df2, low_columns_to_take_full

def model_prep(df, columns_to_take):
    """

    Parameters
    ----------------
    df: - takes transformed dataframe from 'data_prep_columns'

    columns_to_take: - columns you want to input (can be from 'data_prep_columns')
    or chosen in a different way

    Returns
    ----------------
    X_train, X_test, y_train, y_test as expected from train_test_split

    X: is a dataframe with only genes as columns
    """
    data = df.iloc[:, columns_to_take]
    y = df.loc[data.index, 'Low Risk']
    data['label'] = y.copy()
    
    holdout = data.sample(frac=.2, random_state=8)
    train = data.drop(holdout.index).copy()
    X = train.iloc[:, :-1]
    y = train['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.33, random_state=8)
    return X_train, X_test, y_train, y_test, holdout, X

def model_prep_loc(df, columns_to_take):
    data = df.loc[:, columns_to_take]
    y = df.loc[data.index, 'Low Risk']
    data['label'] = y.copy()

    holdout = data.sample(frac=.2, random_state=8)
    train = data.drop(holdout.index).copy()
    X = train.iloc[:, :-1]
    y = train['label']

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.33, random_state=8)
    return X_train, X_test, y_train, y_test, holdout, X

