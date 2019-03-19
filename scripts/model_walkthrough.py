#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:42:57 2019

@author: NCBI Hackathon members and Anne French
"""

import numpy as np
import pandas as pd
import csv
from itertools import zip_longest
from clean import manifest_clinical_merge
from clean import assay_transpose, assay_clinical_merge
from model_comp import data_prep_columns, model_prep, model_comp, model_prep_loc, ensemble_comp
from sklearn.linear_model import LogisticRegression

def find_indices(lst, lst2):
    return [i for i, elem in enumerate(lst) if elem in lst2]

def read_data():
    manifest_df = pd.read_csv('../Manifest_Data/GCD_TARGET_Data_Manifest_AML_NBL_WT_RT.csv')
    aml_disc_df = pd.read_excel('../Clinical_Data/TARGET_AML_ClinicalData_20160714.xlsx')
    AML_df = manifest_clinical_merge(manifest_df, aml_disc_df, 'TARGET-AML')
    assay_df = pd.read_csv('../Expn_Data/TARGET_NBL_AML_RT_WT_TMMCPM_log2_Norm_Counts.csv.zip')
    assay_t_df = assay_transpose(assay_df)
    AML_genes = assay_clinical_merge(assay_t_df, AML_df)
    return AML_genes

def run_classification(gene_df, title, columns_to_take):
    """Data helper for building dataframe and classification.
       
       Columns_to_take can equal 'Max', 'Min', or 'Random'.
    """
    df, a = data_prep_columns(gene_df, columns_to_take)
    
    if columns_to_take == 'Random':
        random_cols = np.random.choice(21404, 2000)
        random_set = set(random_cols)
        random_cols = list(random_set)
        a = [x + 84 for x in random_cols]
        
    X_train, X_test, y_train, y_test, holdout, X = model_prep(df, a)
    xg, rf, ada = model_comp(X_train, X_test, y_train, y_test, title)
    ensemble_comp(X_train, X_test, y_train, y_test, title)
    return X, xg, rf, ada

def classify(df, columns, title):
    """
    Helper method for classification to reduce code.
    """
    df, __ = data_prep_columns(df, 'neither')
    X_train, X_test, y_train, y_test, holdout, X = model_prep_loc(df, columns)
    ensemble_comp(X_train, X_test, y_train, y_test, title)
    
def write_csv(set_lasso, set_xg, set_rf, set_ada, lasso_set2):
    int_original_three = set.intersection(set_lasso, set_xg, set_rf)
    int_first_four = set.intersection(set_lasso, set_xg, set_rf, set_ada)
    int_lasso2_xg = set.intersection(lasso_set2, set_xg)
    int_lasso2_ada = set.intersection(lasso_set2, set_ada)
    int_xg_ada = set.intersection(set_xg, set_ada)
    int_lasso2_xg_ada = set.intersection(lasso_set2, set_xg, set_ada)
    d = [set_lasso, set_xg, set_rf, set_ada, int_original_three, int_first_four, lasso_set2, int_lasso2_xg, int_lasso2_ada, int_xg_ada, int_lasso2_xg_ada]
    csv_rows = zip_longest(*d, fillvalue='')
    with open('important_genes.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(("Lasso Genes 1", "XGBoost Genes", "Random Forest Genes", "AdaBoost Genes", "Intersection Lasso-XG-RF", "Intersection Lasso-XG-RF-Ada", "Lasso Convergence", "Intersection Lasso Convergence-XG", "Intersection Lasso Convergence-Ada", "Intersection XG-Ada", "Intersection Lasso Convergence-XG-Ada"))
        wr.writerows(csv_rows)
    return int_lasso2_xg, int_lasso2_ada, int_xg_ada, int_lasso2_xg_ada

def get_lasso_features(df, convergence='complete'):
    """
    Gets important Lasso features.
    
    Change convergence to 'incomplete' for incomplete convergence.
    """
    df, __ = data_prep_columns(df, 'neither')
    lasso_columns = df.iloc[:, 84:-4].columns
    X_train, X_test, y_train, y_test, holdout, X = model_prep_loc(df, lasso_columns)
    
    log_model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
    if(convergence=='incomplete'):
        log_model = LogisticRegression(penalty='l1', solver='saga')
    
    log_model.fit(X_train, y_train)
    log_coefs = np.array(log_model.coef_)
    mask = log_coefs > 0
    important_lasso_genes = lasso_columns[mask[0]]
    return important_lasso_genes
    
def main():
    pd.options.mode.chained_assignment = None 
    pd.set_option('display.max_columns', 100)
    
    AML_genes = read_data()

    X, xg1, rf1, ada1 = run_classification(AML_genes, "High Variance", 'Max')
    X, xg2, rf2, ada2 = run_classification(AML_genes, "Low Variance", 'Min')
    X, xg3, rf3, ada3 = run_classification(AML_genes, "Random Gene Selection", 'Random')

    #---------Lasso Logistic Regression using all columns---------------
    important_lasso_genes = get_lasso_features(AML_genes, convergence='incomplete')

    #-----------Tree Performance Using Lasso Genes------------
    title = "Performance using lasso genes"
    df, __ = data_prep_columns(AML_genes, 'neither')
    X_train, X_test, y_train, y_test, holdout, X = model_prep_loc(df, important_lasso_genes)
    xg_lass, rf_lass, ada_lass = model_comp(X_train, X_test, y_train, y_test, title)
    ensemble_comp(X_train, X_test, y_train, y_test, title)

    #-------Feature Importance When Running Tree Over All Features-----
    title = "Performance using all features"
    all_columns = df.iloc[:, 84:-4].columns
    df, __ = data_prep_columns(AML_genes, 'neither')
    X_train, X_test, y_train, y_test, holdout, X = model_prep_loc(df, all_columns)
    xg_all, rf_all, ada_all = model_comp(X_train, X_test, y_train, y_test)
    ensemble_comp(X_train, X_test, y_train, y_test, title)

    xg_all_important = X.columns[np.array(xg_all.feature_importances_) > 0]
    rf_all_important = X.columns[np.array(rf_all.feature_importances_) > 0]
    ada_all_important = X.columns[np.array(ada_all.feature_importances_) > 0]

    #-------Four sets of important features so far---------------------
    set_lasso = set(important_lasso_genes)
    set_xg = set(xg_all_important)
    set_rf = set(rf_all_important)
    set_ada = set(ada_all_important)

    #--------Lasso Complete Convergence--------------------        
    important_lasso_genes2 = get_lasso_features(AML_genes)
    lasso_set2 = set(important_lasso_genes2)

    int_lasso2_xg, int_lasso2_ada, int_xg_ada, int_lasso2_xg_ada = write_csv(set_lasso, set_xg, set_rf, set_ada, lasso_set2)

    #--------Final Performance Tests--------------
    classify(AML_genes, important_lasso_genes2, "Performance using converged Lasso genes")
    classify(AML_genes, set_xg, "Performance using XGBoost genes")
    classify(AML_genes, set_rf, "Performance using Random Forest genes")
    classify(AML_genes, set_ada, "Performance using AdaBoost genes")
    classify(AML_genes, int_lasso2_xg, "Performance using intersection of converged Lasso and XGBoost genes")
    classify(AML_genes, int_lasso2_ada, "Performance using intersection of converged Lasso and AdaBoost genes")
    classify(AML_genes, int_xg_ada, "Performance using intersection of XGBoost and AdaBoost genes")
    classify(AML_genes, int_lasso2_xg_ada, "Performance using intersection of converged Lasso, XGBoost, and AdaBoost genes")
    
if __name__ == '__main__':
   main()