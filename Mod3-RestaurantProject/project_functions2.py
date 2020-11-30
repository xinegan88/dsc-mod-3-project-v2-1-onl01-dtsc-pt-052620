import numpy as np
import pandas as pd

import sklearn.model_selection as ms
import sklearn.metrics as mx
import sklearn.ensemble as en

import matplotlib.pyplot as plt
import seaborn as sns

import json
import requests

## yelp data functions

def get_businesses(location, term, api_key):
    headers = {'Authorization': 'Bearer %s' % api_key}
    url = 'https://api.yelp.com/v3/businesses/search'

    data = []
    for offset in range(0, 1000, 50):
        params = {
            'limit': 50, 
            'location': location.replace(' ', '+'),
            'term': term.replace(' ', '+'),
            'offset': offset
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data += response.json()['businesses']
        elif response.status_code == 400:
            print('400 Bad Request')
            break
            
    print('Obtained ', term, ' data.')

    return data


def clean_yelp_df(df):
    
    # hide some columns
    hide = ['name', 'id', 'alias', 'image_url', 'url', 'is_closed', 'display_phone',
           'phone']
    df = df.drop(hide, axis=1)
    
    
    # clean up categories
    df['cat_1'] = df.categories.apply(lambda x: x[0]['alias'])
    df['cat_2'] = df.categories.apply(lambda x: x[1]['alias'] if len(x) > 1 else 0)
    df['cat_3'] = df.categories.apply(lambda x: x[2]['alias'] if len(x) > 2 else 0)
    
    dummy1 = pd.get_dummies(df.cat_1).add_suffix('_cat_1')
    dummy2 = pd.get_dummies(df.cat_2).add_suffix('_cat_2')
    dummy3 = pd.get_dummies(df.cat_3).add_suffix('_cat_3')
    all_dummies = pd.concat([dummy1, dummy2, dummy3], axis=1)    
    df = pd.concat([df.drop(['cat_1', 'cat_2', 'cat_3', 'categories'], 
                            axis=1), all_dummies], axis=1)
    if '0_cat_2' in list(df.columns):
        df = df.drop('0_cat_2', axis=1)
    
    if '0_cat_3' in list(df.columns):
        df = df.drop('0_cat_3', axis=1)
        
    # clean up coordinates
    df['lat'] = df.coordinates.apply(lambda x: x['latitude'])
    df['long'] = df.coordinates.apply(lambda x: x['longitude'])
    df = df.drop('coordinates', axis=1)
    
    # clean up transactions
    df['delivery'] = df.transactions.apply(lambda x: 1 if 'delivery' in x else 0)
    df['pickup'] = df.transactions.apply(lambda x: 1 if 'pickup' in x else 0)
    df['reservation'] = df.transactions.apply(lambda x: 1 if 'reservation' in x else 0)
    df = df.drop('transactions', axis=1)
    
    # clean up price
    df['price'] = df.price.apply(lambda x: len(str(x)))
    price_avg = df.price.mean()
    df['price'] = df.price.fillna(price_avg)
    
    dummies = pd.get_dummies(df.price).add_prefix('price_')
    df = pd.concat([df.drop('price', axis=1), dummies], axis=1)
 
    # clean up location
    loc_keys = list(df.location.iloc[0].keys())
    num = 0
    for k in loc_keys:
        df[k] = df.location.apply(lambda x: x[k])
     
    df = df.drop(['location', 'address2', 'address3', 'state', 'country'], axis=1)
        
    # establishing a target
    df['above_avg'] = df.rating.apply(lambda x: 1 if x > 4 else 0)
    df = df.drop('rating', axis=1)
    
    return df

def plot_my_conf_matrix(cm, ax):
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, cmap='Blues_r', ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('Actual Label', fontsize=10)
    ax.set_title('Confusion Matrix',fontsize=12)
    
    
def plot_my_roc_curve(clf, X_test, y_test, ax):
    mx.plot_roc_curve(clf, X_test, y_test, alpha=1, lw=2, ax=ax)
    ax.set_title('ROC Curve/AUC Score',fontsize=12)
    
    
def base_model(clf, X, y, test_size, random_state):

    print('\nBasic Model with Train, Test, Split for:', clf, '\n')
    
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=test_size, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(mx.classification_report(y_test,y_pred))
    cm = mx.confusion_matrix(y_test, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))
    plot_my_conf_matrix(cm, ax1)
    plot_my_roc_curve(clf, X_test, y_test, ax2)
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    plt.show()
    
    
def plot_my_cross_val_roc_curve(clf, X, y, cv):
    
    
    fig1 = plt.figure(figsize=[8, 12])
    ax1 = fig1.add_subplot(111, aspect='equal')

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)

    i = 1
    for train,test in cv.split(X,y):
        prediction = clf.fit(X.iloc[train],y.iloc[train]).predict_proba(X.iloc[test])
        fpr, tpr, t = mx.roc_curve(y[test], prediction[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = mx.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = mx.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    

def cv_model_scores(clf, X, y, cv, num):
    
    scores = pd.DataFrame()
    scores['Accuracy'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    scores['AUC'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    scores['Precision'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='precision')
    scores['Recall'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='recall')
    scores['F1'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='f1')
    
        
    scores = scores.sort_values(by='Precision').reset_index().drop('index', axis=1)
    scores.loc['Model '+ str(num)] = scores.mean()
    
    return scores.loc['Model '+str(num)]


def run_cross_val_model(clf, X, y, cv):

    scores = pd.DataFrame()
    scores['Accuracy'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    scores['AUC'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    scores['Precision'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='precision')
    scores['Recall'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='recall')
    scores['F1'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='f1')
    
        
    scores = scores.sort_values(by='Precision').reset_index().drop('index', axis=1)
    scores.loc['mean'] = scores.mean()
    
    print('\nCross Validation Model with Repeated Stratified K-Fold for:', clf, '\n')
    
    plot_my_cross_val_roc_curve(clf, X, y, cv)
    
    print('\nScores for:', clf)
    display(scores)

    print('=========================================================================================================')
    
### smote functions
import imblearn.over_sampling as imbos

def base_model_smote(clf, X, y, test_size, random_state):

    print('\nBasic Model with Train, Test, Split with SMOTE for:', clf, '\n')
    
    smote = imbos.SMOTE(random_state=1)
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, y_train = smote.fit_sample(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(mx.classification_report(y_test,y_pred))
    print(mx.accuracy_score(y_test, y_pred))

    cm = mx.confusion_matrix(y_test, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))
    plot_my_conf_matrix(cm, ax1)
    plot_my_roc_curve(clf, X_test, y_test, ax2)
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    plt.show()


def plot_my_cross_val_roc_curve_smote(clf, X, y, cv):
    
    smote = imbos.SMOTE(random_state=1)
    
    fig1 = plt.figure(figsize=[12,8])
    ax1 = fig1.add_subplot(111, aspect='equal')

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)

    i = 1
    for train,test in cv.split(X,y):
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        
        X_train, y_train = smote.fit_sample(X_train, y_train)
          
        prediction = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, t = mx.roc_curve(y_test, prediction[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = mx.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = mx.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
def run_cross_val_model_smote(clf, X, y, cv):

    scores = pd.DataFrame()
    scores['Accuracy'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    scores['AUC'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    scores['Precision'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='precision')
    scores['Recall'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='recall')
    scores['F1'] = ms.cross_val_score(clf, X, y, cv=cv, scoring='f1')
    
        
    scores = scores.sort_values(by='Precision').reset_index().drop('index', axis=1)
    scores.loc['mean'] = scores.mean()
    
    print('\nCross Validation Model with Repeated Stratified K-Fold and SMOTE for:', clf, '\n')
    
    plot_my_cross_val_roc_curve_smote(clf, X, y, cv)
    
    print('\nScores for:', clf)
    display(scores)

    print('=========================================================================================================')
    
def feature_importance(clf, importances, X):

    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    cols = pd.DataFrame()
    num = 0
    for col in list(X.columns):
        cols[col] = num
        num + 1
    
    cols = cols.T
#     cols = cols.reset_index()
    cols = cols.reset_index().to_dict()['index']
    
    imp_ind = pd.DataFrame()
    imp_ind['Idx'] = indices
    imp_ind['Importance'] = importances
    imp_ind['Feature'] = imp_ind.Idx.map(cols)
    imp_ind['Importance'] = imp_ind.Importance.apply(lambda x: round(x, 4))
    imp_ind = imp_ind.sort_values(by='Importance', ascending=False).reset_index().drop('index', axis=1)
    display(imp_ind[:15])

    palette = sns.color_palette('Blues_r', 20, 1)
    sns.set_theme(context='notebook', style='white')
    
    ax = sns.barplot(x='Feature', y='Importance', data=imp_ind[:15],
                palette=palette);
    ax.set_xlabel('Features',fontsize=15);
    ax.set_ylabel('Importance',fontsize=15);
    ax.set_ylim(0, 0.2)
    ax.set_xticklabels(labels=imp_ind.Feature[:15], rotation=90, fontsize=12)
    plt.title('Top 10 Features', fontsize=18);