# -*- coding: utf-8 -*-
# In[import]

import os, io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from joblib import dump, load
from sklearn.metrics import (classification_report,confusion_matrix, make_scorer, accuracy_score, f1_score,
                             precision_recall_fscore_support, ConfusionMatrixDisplay, roc_auc_score,  
                             average_precision_score,  roc_curve,  auc,  precision_recall_curve, get_scorer)

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from numpy import interp, sqrt, argmax
from pandas import read_csv
from numpy import interp
from math import floor
from imblearn.pipeline import Pipeline as imblearn_Pipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier,  plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, SVMSMOTE, ADASYN
#from imblearn.pipeline import Pipeline, make_pipeline
from datetime import timedelta, timezone, datetime
import time
import joblib

from scipy import sparse 
from scipy.stats import uniform, randint

# In[]
# function

def custom_classify(probabilities, thresholds):
    # Applying the thresholds for each class
    classified = (probabilities > thresholds).astype(int)
    exceedance = (probabilities - thresholds)/thresholds  #用超過的比例, 而不適超過的絕對值
    # Modified section for handling multiple classes exceeding the threshold
    results = []
    for i in range(len(classified)):
        row = classified[i]
        if row.sum() > 1:  # More than one class exceeds the threshold
            # Limit exceedance to the classes that exceeded the threshold
            valid_exceedance = exceedance[i] * row
            # Find the class with the maximum exceedance
            max_exceedance_index = np.argmax(valid_exceedance)
            class_output = np.zeros_like(row)
            class_output[max_exceedance_index] = 1
            results.append(class_output)
        else:
            results.append(row.astype(int))
    
    return np.array(results)

def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)  # [class0, class1, class2]
    FN = cm.sum(axis=1) - np.diag(cm)  # [class0, class1, class2]
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

   # Calculate class-wise performance
    class_metrics = {}
    num_classes = cm.shape[0]
    for i in range(num_classes):
        class_accuracy = (TP[i] + TN[i]) / (TP[i] + FP[i] + FN[i] + TN[i])
        class_sensitivity = TP[i] / (TP[i] + FN[i])
        class_specificity = TN[i] / (TN[i] + FP[i])
        class_ppv = TP[i] / (TP[i] + FP[i])
        class_npv = TN[i] / (TN[i] + FN[i])
        class_f1 = f1_score(y_true == i, y_pred == i)
       
        class_metrics[f'Class {i}'] = {
            'Accuracy': class_accuracy,
            'F1 Score': class_f1,
            'Sensitivity': class_sensitivity,
            'Specificity': class_specificity,
            'PPV': class_ppv,
            'NPV': class_npv
        }

    return class_metrics, cm


def PlotConfusionMatrix(cm):
    matrix = cm
    fs = 30
    plt.figure(figsize=(7, 4))
    sns.heatmap(matrix, annot=True, fmt="d",annot_kws={"size": fs})
    plt.title('Confusion matrix', fontsize=fs)
    # plt.xticks([i+0.5 for i in range(len(label_names))], label_names)
    # plt.yticks([i+0.5 for i in range(len(label_names))], label_names)
    plt.xlabel("Predict", fontsize=fs)
    plt.ylabel("Ground Truth", fontsize=fs)
    plt.savefig(f"figure/confusion_matrix.png", dpi = 300, transparent = True, bbox_inches = "tight")
    plt.show()
    
    
def imputer(df, train = True, mean = None):
    if train:
        mean = df.mean()
        return df.fillna(mean), mean
    else:
        return df.fillna(mean)



def scaling (x, scaler, train= True):
    if train:
        scaler = StandardScaler()
        x_scaler= scaler.fit(x)
        x_scale = x_scaler.transform(x)
        x_scale= pd.DataFrame(x_scale, columns = x.columns)

    else:
        x_scaler=scaler
        x_scale = x_scaler.transform(x)
        x_scale= pd.DataFrame(x_scale, columns = x.columns)

    return x_scale ,x_scaler
    
# In[parameter setting]
    
random_seed= 564
class_weights = {0: 1, 1: 1, 2: 1}
'''
class_1_threshold= 0.074
class_2_threshold= 0.017
class_0_threshold= 0.91
'''

class_1_threshold= 0.037
class_2_threshold= 0.023
class_0_threshold= 0.940

thresholds = [class_0_threshold, class_1_threshold, class_2_threshold]
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)  # None 表示显示所有行
pd.set_option('display.max_columns', None)

data_date='20240208 CLSI'
data_path= r'D:\張裕鑫\RESEARCH\CPD\急診預測\格蘭氏染色'
version_date= 20240920

result_path=rf'D:\python_code\GS\result\performance\{version_date}'
variable_path=rf'D:\python_code\GS\variable\training\{version_date}'
feature_path=rf'D:\python_code\GS\variable\feature\{version_date}'


if not os.path.exists(result_path):
    os.makedirs(result_path)
    
if not os.path.exists(variable_path):
    os.makedirs(variable_path)
    
if not os.path.exists(feature_path):
    os.makedirs(feature_path) 
# In[Import data]
    
    
data = pd.read_csv(fr'{data_path}\{data_date}\CMUH 2021_2022\CMUH_GS_2021.csv')

#將汙染菌歸入陽性
#data = pd.read_csv(r'D:\張裕鑫\RESEARCH\CPD\急診預測\格蘭氏染色\20231221 將汙染菌歸入陽性\CMUH 2021_2022\CMUH_202108_202212_12_hours.csv')

data_cmuh= pd.read_csv(fr'{data_path}\{data_date}\CMUH 2023\CMUH_GS_2023.csv')
data_WK = pd.read_csv(fr'{data_path}\{data_date}\WK\WK_GS_2023.csv')
data_AN = pd.read_csv(fr'{data_path}\{data_date}\AN\AN_GS_2023.csv')

#data= data_origin.copy()


# [將特徵和標籤分開]
y = data.iloc[:, -1]
x = data.iloc[:,6:93]
# 指定x,y
y_cmuh = data_cmuh.iloc[:,-1]
x_cmuh = data_cmuh.iloc[:,6:93]

y_wk = data_WK.iloc[:,-1]
x_wk = data_WK.iloc[:,6:93]


y_an = data_AN.iloc[:,-1]
x_an = data_AN.iloc[:,6:93]

# drop SD_V_MO 
'''
x.drop(columns='SD_V_MO',inplace=True)
x_cmuh.drop(columns='SD_V_MO',inplace=True)
x_wk.drop(columns='SD_V_MO',inplace=True)
x_an.drop(columns='SD_V_MO',inplace=True)
'''

# In[]
from collections import Counter

for i in [y, y_cmuh, y_wk, y_an] :
    element_counts = Counter(i)
    total_elements = len(i)
    # 打印结果
    print(f"total case: {total_elements}")
    print("Counts of 0:", element_counts[0])
    print("Counts of 1:", element_counts[1])
    print("Counts of 2:", element_counts[2])


    # 计算各个元素的比例
    element_proportions = {key: count / total_elements for key, count in element_counts.items()}
    
    # 打印各个元素的比例
    for key, proportion in element_proportions.items():
        print(f"Proportion of {key}: {proportion:.2%}")



# In[start training]
total_start_time = time.time()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=random_seed)
results = pd.DataFrame(columns=['Model', 'AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])


x_train_imp, m_mean = imputer(x_train, train=True)
x_test_imp = imputer(x_test, False, m_mean)



x_train_scale, train_scaler = scaling(x_train_imp, None, train=True)
x_test_scale, train_scaler = scaling (x_test_imp, train_scaler, train= False)

#joblib.dump(scaler, f"scaler")

#X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scale, y_train)
X_train_resampled, y_train_resampled= x_train_scale, y_train
# model fit

#model= LGBMClassifier()
#model= XGBClassifier()
#model= RandomForestClassifier()
#model= LogisticRegression()
model= CatBoostClassifier(verbose=False, class_weights=class_weights)
#model= SVC(gamma='auto',probability=True)

model.fit(X_train_resampled, y_train_resampled)

#y_pred = model.predict(X_test_scale)
y_prob = model.predict_proba(x_test_scale)
y_prob_bin = custom_classify(y_prob, thresholds)
y_pred = y_prob_bin.argmax(axis=1)
#y_prob_bin = label_binarize(y_prob.argmax(axis=1), classes=[0, 1, 2])


# 在這裡計算每個類別的AUROC和AUPRC
auroc = roc_auc_score(y_test_bin, y_prob, average=None)
auprc = average_precision_score(y_test_bin, y_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
# metrics, cm_model = get_metrics(y_test, y_pred)
metrics, cm_model = get_metrics(y_test, y_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_df = pd.DataFrame(metrics).T
class_metrics_df['AUROC'] = auroc
class_metrics_df['AUPRC'] = auprc
results = pd.concat([results, class_metrics_df])

print (results)
total_end_time = time.time() # 記錄結束時間
total_elapsed_time = total_end_time - total_start_time  # 計算所花費的時間
print(f"total took {total_elapsed_time:.2f} seconds to run.")
# In[]

# model = joblib.load(f'model')

importances = model.feature_importances_
indices = np.argsort(importances) [::][-15:]
features = x.columns
plt.subplots(figsize=(6,18))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"rf_feature_importance.png", dpi = 300, facecolor='white', transparent = True, bbox_inches = "tight")
plt.show()




# In[] Feature selection


# 將特徵和標籤分開
'''
x = data.iloc[:,6:93]
y = data.iloc[:, -1]
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=random_seed)
# 指定x,y
'''
y_cmuh = data_cmuh.iloc[:,-1]
x_cmuh = data_cmuh.iloc[:,6:93]

y_wk = data_WK.iloc[:,-1]
x_wk = data_WK.iloc[:,6:93]

y_an = data_AN.iloc[:,-1]
x_an = data_AN.iloc[:,6:93]
'''
importances = model.feature_importances_
indices = np.argsort(importances)

dump(importances, rf'{feature_path}\importances.joblib')
dump(indices, rf'{feature_path}\index.joblib')


features = x.columns.tolist()
feature_sel = [features[i] for i in indices]
features_selected = ['0'] + feature_sel

results = pd.DataFrame(columns=[ 'AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
results_cmuh = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
results_wk = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
results_an = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])


for i in tqdm(features_selected[:-1]):
    if i == '0':
        features_selected = feature_sel.copy()
    else:
        features_selected.remove(i)

    x_train_sel = x_train[features_selected]
    x_test_sel = x_test[features_selected]
    x_cmuh_sel = x_cmuh[features_selected]
    x_wk_sel = x_wk[features_selected]
    x_an_sel = x_an[features_selected]

    
    
    x_train_imp, m_mean = imputer(x_train_sel, train=True)
    x_test_imp = imputer(x_test_sel, False, m_mean)
    x_cmuh_imp = imputer(x_cmuh_sel, False, m_mean)
    x_wk_imp = imputer(x_wk_sel, False, m_mean)
    x_an_imp = imputer(x_an_sel, False, m_mean)
    
    
    
    x_train_scale, train_scaler = scaling(x_train_imp, None, train=True)
    x_test_scale, train_scaler = scaling (x_test_imp, train_scaler, train= False)
    x_cmuh_scale, train_scaler = scaling (x_cmuh_imp, train_scaler, train= False)
    x_wk_scale, train_scaler = scaling (x_wk_imp, train_scaler, train= False)
    x_an_scale, train_scaler = scaling (x_an_imp, train_scaler, train= False)
    
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_cmuh_bin = label_binarize(y_cmuh, classes=[0, 1, 2])
    y_wk_bin = label_binarize(y_wk, classes=[0, 1, 2])
    y_an_bin = label_binarize(y_an, classes=[0, 1, 2])
    
     ## Internal ##
    
    #X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scale, y_train)
    x_train_resampled, y_train_resampled= x_train_scale, y_train
    # model fit
    
    #model= LGBMClassifier()
    model= CatBoostClassifier(verbose=False)
    model.fit(x_train_resampled, y_train_resampled)
    #joblib.dump(model, f"model")
    
    y_prob = model.predict_proba(x_test_scale)
    y_prob_bin = custom_classify(y_prob, thresholds)
    y_pred = y_prob_bin.argmax(axis=1)
    
    # 在這裡計算每個類別的AUROC和AUPRC
    auroc = roc_auc_score(y_test_bin, y_prob, average=None)
    auprc = average_precision_score(y_test_bin, y_prob, average=None)
    
    # 將每個類別的AUROC和AUPRC存儲在列表中
    metrics, cm = get_metrics(y_test, y_pred)
    
    # 将class_metrics字典转换为DataFrame
    class_metrics_df = pd.DataFrame(metrics).T
    class_metrics_df['AUROC'] = auroc
    class_metrics_df['AUPRC'] = auprc
    results = pd.concat([results, class_metrics_df])

    ## External ## cmuh

    y_cmuh_prob = model.predict_proba(x_cmuh_scale)
    y_cmuh_prob_bin = custom_classify(y_cmuh_prob, thresholds)
    y_cmuh_pred = y_cmuh_prob_bin.argmax(axis=1)
    
    
    
    # 在這裡計算每個類別的AUROC和AUPRC
    auroc_cmuh = roc_auc_score(y_cmuh_bin, y_cmuh_prob, average=None)
    auprc_cmuh = average_precision_score(y_cmuh_bin, y_cmuh_prob, average=None)
    
    # 將每個類別的AUROC和AUPRC存儲在列表中
    metrics_cmuh, cm_cmuh = get_metrics(y_cmuh, y_cmuh_pred)
    
    # 将class_metrics字典转换为DataFrame
    class_metrics_cmuh = pd.DataFrame(metrics_cmuh).T
    class_metrics_cmuh['AUROC'] = auroc_cmuh
    class_metrics_cmuh['AUPRC'] = auprc_cmuh
    results_cmuh = pd.concat([results_cmuh, class_metrics_cmuh])

    ## External ## wk

    y_wk_prob = model.predict_proba(x_wk_scale)
    y_wk_prob_bin = custom_classify(y_wk_prob, thresholds)
    y_wk_pred = y_wk_prob_bin.argmax(axis=1)
    
    
    # 在這裡計算每個類別的AUROC和AUPRC
    auroc_wk = roc_auc_score(y_wk_bin, y_wk_prob, average=None)
    auprc_wk = average_precision_score(y_wk_bin, y_wk_prob, average=None)
    
    # 將每個類別的AUROC和AUPRC存儲在列表中
    metrics_wk, cm_wk = get_metrics(y_wk, y_wk_pred)
    
    # 将class_metrics字典转换为DataFrame
    class_metrics_wk = pd.DataFrame(metrics_wk).T
    class_metrics_wk['AUROC'] = auroc_wk
    class_metrics_wk['AUPRC'] = auprc_wk
    results_wk = pd.concat([results_wk, class_metrics_wk])
    
    
    
    ## External ## an
    y_an_prob = model.predict_proba(x_an_scale)
    y_an_prob_bin = custom_classify(y_an_prob, thresholds)
    y_an_pred = y_an_prob_bin.argmax(axis=1)


    # 在這裡計算每個類別的AUROC和AUPRC
    auroc_an = roc_auc_score(y_an_bin, y_an_prob, average=None)
    auprc_an = average_precision_score(y_an_bin, y_an_prob, average=None)
    
    # 將每個類別的AUROC和AUPRC存儲在列表中
    metrics_an, cm_an = get_metrics(y_an, y_an_pred)
    
    # 将class_metrics字典转换为DataFrame
    class_metrics_an = pd.DataFrame(metrics_an).T
    class_metrics_an['AUROC'] = auroc_an
    class_metrics_an['AUPRC'] = auprc_an
    results_an = pd.concat([results_an, class_metrics_an])
    

# In[]
#先把class從index中拿掉, 自己當作一欄
results['class'] = results.index  
fs = results.reset_index(drop=True)
#加入移除的Feature list
fs.index=fs.index//3


feature_removed=  [''] + feature_sel[:-1]
repeated_feature_removed = []
for item in feature_removed:
    repeated_feature_removed .extend([item] * 3)


fs['feature_removed'] = repeated_feature_removed
fs.tail()
fs[fs.AUROC==fs.AUROC.max()]
fs[fs.AUROC >= 0.853]

dump(fs, rf'{feature_path}\CMUH_2021_feature.joblib')
# In[]

results_cmuh['class'] = results_cmuh.index
fs_cmuh = results_cmuh.reset_index(drop=True)
fs_cmuh.index=fs_cmuh.index//3

fs_cmuh['feature_removed'] = repeated_feature_removed
fs_cmuh.tail()
fs_cmuh[fs_cmuh.AUROC == fs_cmuh.AUROC.max()]
fs_cmuh[(fs_cmuh.AUROC >= 0.860) & (fs_cmuh['class']=='Class 1') & (fs_cmuh['AUPRC']>= 0.40)]

dump(fs_cmuh, rf'{feature_path}\CMUH_2023_feature.joblib')
# In[]

results_wk['class'] = results_wk.index
fs_wk = results_wk.reset_index(drop=True)
fs_wk.index=fs_wk.index//3


fs_wk['feature_removed'] = repeated_feature_removed
fs_wk.tail()
fs_wk[(fs_wk.AUROC>= 0.855) & (fs_wk['class']=='Class 1') & (fs_wk['AUPRC']>= 0.40)]


dump(fs_wk, rf'{feature_path}\WK_feature.joblib')



# In[]

results_an['class'] = results_an.index
fs_an = results_an.reset_index(drop=True)
fs_an.index=fs_an.index//3


fs_an['feature_removed'] = repeated_feature_removed
fs_an.tail()
fs_an[(fs_an.AUROC>= 0.855) & (fs_an['class']=='Class 1') & (fs_an['AUPRC']>= 0.40)]


dump(fs_an, rf'{feature_path}\AN_feature.joblib')

# In[合併]

# 合并所有表格
combined_feature_select = pd.concat([fs, fs_cmuh, fs_wk, fs_an], axis=1)

# 保存合并后的表格
combined_feature_select.to_csv(rf'{feature_path}\feature_sel.csv', index=True, encoding='utf_8_sig')
dump(combined_feature_select, rf'{feature_path}\feature_sel.joblib')



# In[chose bet feature number]

feature_select=    load( rf'{feature_path}\CMUH_2021_feature.joblib')
feature_select_test=    load( rf'{feature_path}\CMUH_2023_feature.joblib')
feature_select_WK= load (rf'{feature_path}\WK_feature.joblib')
feature_select_AN= load (rf'{feature_path}\AN_feature.joblib')

importances = load(rf'{feature_path}\importances.joblib')
indices =  load(rf'{feature_path}\index.joblib')

selected_model_index= 28




outcomes = pd.DataFrame(columns=[ 'AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
outcomes_cmuh = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
outcomes_wk = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
outcomes_an = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

features = x.columns.tolist()
feature_sel = [features[i] for i in indices]
best_features = feature_sel[selected_model_index:]
print (best_features)
n = len(best_features)

x_train_best = x_train[best_features]
x_test_best = x_test[best_features]
x_cmuh_best = x_cmuh[best_features]
x_wk_best = x_wk[best_features]
x_an_best = x_an[best_features]

x_train_imp, m_mean = imputer(x_train_best, train=True)
x_test_imp = imputer(x_test_best, False, m_mean)
x_cmuh_imp = imputer(x_cmuh_best, False, m_mean)
x_wk_imp = imputer(x_wk_best, False, m_mean)
x_an_imp = imputer(x_an_best, False, m_mean)



x_train_scale, train_scaler = scaling(x_train_imp, None, train=True)
x_test_scale, train_scaler = scaling (x_test_imp, train_scaler, train= False)
x_cmuh_scale, train_scaler = scaling (x_cmuh_imp, train_scaler, train= False)
x_wk_scale, train_scaler = scaling (x_wk_imp, train_scaler, train= False)
x_an_scale, train_scaler = scaling (x_an_imp, train_scaler, train= False)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_cmuh_bin = label_binarize(y_cmuh, classes=[0, 1, 2])
y_wk_bin = label_binarize(y_wk, classes=[0, 1, 2])
y_an_bin = label_binarize(y_an, classes=[0, 1, 2])


 ## Internal ##
#X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scale, y_train)
x_train_resampled, y_train_resampled= x_train_scale, y_train
# model fit

#model= LGBMClassifier()
model= CatBoostClassifier(verbose=False)
model.fit(x_train_resampled, y_train_resampled)
#joblib.dump(model, f"model")

y_prob = model.predict_proba(x_test_scale)
y_prob_bin = custom_classify(y_prob, thresholds)
y_pred = y_prob_bin.argmax(axis=1)

# 在這裡計算每個類別的AUROC和AUPRC
auroc = roc_auc_score(y_test_bin, y_prob, average=None)
auprc = average_precision_score(y_test_bin, y_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics, cm = get_metrics(y_test, y_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_df = pd.DataFrame(metrics).T
class_metrics_df['AUROC'] = auroc
class_metrics_df['AUPRC'] = auprc
outcomes = pd.concat([outcomes, class_metrics_df])

## External ## cmuh

y_cmuh_prob = model.predict_proba(x_cmuh_scale)
y_cmuh_prob_bin = custom_classify(y_cmuh_prob, thresholds)
y_cmuh_pred = y_cmuh_prob_bin.argmax(axis=1)



# 在這裡計算每個類別的AUROC和AUPRC
auroc_cmuh = roc_auc_score(y_cmuh_bin, y_cmuh_prob, average=None)
auprc_cmuh = average_precision_score(y_cmuh_bin, y_cmuh_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics_cmuh, cm_cmuh = get_metrics(y_cmuh, y_cmuh_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_cmuh = pd.DataFrame(metrics_cmuh).T
class_metrics_cmuh['AUROC'] = auroc_cmuh
class_metrics_cmuh['AUPRC'] = auprc_cmuh
outcomes_cmuh = pd.concat([outcomes_cmuh, class_metrics_cmuh])

## External ## wk

y_wk_prob = model.predict_proba(x_wk_scale)
y_wk_prob_bin = custom_classify(y_wk_prob, thresholds)
y_wk_pred = y_wk_prob_bin.argmax(axis=1)


# 在這裡計算每個類別的AUROC和AUPRC
auroc_wk = roc_auc_score(y_wk_bin, y_wk_prob, average=None)
auprc_wk = average_precision_score(y_wk_bin, y_wk_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics_wk, cm_wk = get_metrics(y_wk, y_wk_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_wk = pd.DataFrame(metrics_wk).T
class_metrics_wk['AUROC'] = auroc_wk
class_metrics_wk['AUPRC'] = auprc_wk
outcomes_wk = pd.concat([outcomes_wk, class_metrics_wk])



## External ## an
y_an_prob = model.predict_proba(x_an_scale)
y_an_prob_bin = custom_classify(y_an_prob, thresholds)
y_an_pred = y_an_prob_bin.argmax(axis=1)


# 在這裡計算每個類別的AUROC和AUPRC
auroc_an = roc_auc_score(y_an_bin, y_an_prob, average=None)
auprc_an = average_precision_score(y_an_bin, y_an_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics_an, cm_an = get_metrics(y_an, y_an_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_an = pd.DataFrame(metrics_an).T
class_metrics_an['AUROC'] = auroc_an
class_metrics_an['AUPRC'] = auprc_an
outcomes_an = pd.concat([outcomes_an, class_metrics_an])


print (f'train set performance= {outcomes}')
print (f'test set performance= {outcomes_cmuh}')
print (f'wk set performance= {outcomes_wk}')
print (f'an set performance= {outcomes_an}')


# In[find best threshold]

fpr, tpr, threshold = roc_curve(y_cmuh, y_cmuh_prob)

# In[threshold point]
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

roc_auc = roc_auc_score(y_ext, prob_ext)
fpr, tpr, thresholds = roc_curve(y_ext, prob_ext,drop_intermediate=False)
plt.plot(thresholds,np.abs(fpr+tpr-1))
plt.xlabel("Threshold")
plt.ylabel("|FPR + TPR - 1|")
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.show()


# In[confusion matrix]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 绘制混淆矩阵
class_names = ['Class 0', 'Class 1', 'Class 2']

# 使用Seaborn绘制热图
sns.set(font_scale=1.2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cmuh, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[] hyperparameter tuning
'''
param_grid = {
    'n_estimators': [500,700,900],
    'learning_rate': [0.025, 0.03, 0.035 ],
    'depth': [8, 10,12],
    'l2_leaf_reg': [1, 3, 5],
    #'border_count': [32, 64, 128],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bylevel': [0.8, 0.9, 1.0]
    #'random_strength': [0.1, 0.5, 1],
    #'scale_pos_weight': [1, 2, 5]
    # Add more hyperparameters and their values as needed

}
'''
total_start_time = time.time()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=random_seed)


best_number= 28

outcomes = pd.DataFrame(columns=[ 'AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
outcomes_cmuh = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
outcomes_wk = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
outcomes_an = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

features = x.columns.tolist()
feature_sel = [features[i] for i in indices]
best_features = feature_sel[best_number:]
print (best_features)
n = len(best_features)

'''
# LGBM
param_distributions = {
    'model__learning_rate': [0.05, 0.1,  0.2, 0.3 ], # default= 0.1
    'model__n_estimators': [50, 100, 150, 200], # default= 100
    'model__num_leaves': [60, 70 ,80, 90, 100],      # default= 31
    'model__boosting_type': [ 'gbdt','dart'],        # default= gbdt
    'model__min_child_samples': [ 10, 15, 20, 30],      # default= 20
    'model__max_depth': [8,10,12]                    # default= -1 (no limit)
}
'''
# CATboost
param_distributions = {
    'model__depth': randint(4, 10),
    'model__learning_rate': uniform(0.005, 0.05),
    'model__iterations': randint(800, 1400),
    'model__l2_leaf_reg': randint(1, 6),
    'model__border_count': randint(1, 255),
    #'model__bagging_temperature': randint(0, 8),
    'model__random_strength': uniform(1, 10)
}

scoring= scoring = {
    'ROC_AUC': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    #'F1_Score': make_scorer(f1_score, average='micro')
    }
#features = x.columns.tolist()
#feature_sel = [features[i] for i in indices]

x_train_best = x_train[best_features]
x_test_best = x_test[best_features]
# Define your pipeline without any preprocessing steps
pipeline = imblearn_Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaling', StandardScaler()),  # Scaling transformer
    ('model' , CatBoostClassifier(verbose=False))  #('model', LGBMClassifier(verbose=0))
])
       
    
# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=1300,  # Adjust as needed
    cv=3,  # Use the cross-validation strategy
    scoring=scoring,
    refit='ROC_AUC',
    verbose=2,
    random_state=random_seed,
    n_jobs=None
)

# Fit the random search on the resampled data
random_search.fit(x_train_best, y_train)


# 在测试集上评估最佳模型
y_pred = random_search.predict(x_test_best)
roc_auc = roc_auc_score(y_test, y_pred,multi_class='ovr')
#f1 = f1_score(y_test, y_pred, average='micro')
print(f"ROC AUC Score on Test Set: {roc_auc:.4f}")
#print(f"F1 Score on Test Set: {f1:.4f}")


# Evaluate the best model on the validation data (x_val_scaled, y_val_cv)
total_end_time = time.time()  # 記錄結束時間
total_elapsed_time = total_end_time - total_start_time  # 計算所花費的時間
print(f"total took {total_elapsed_time:.2f} seconds to run.")


# In[]

print (random_search.best_score_)
print (random_search.best_params_)
print (random_search.best_estimator_)


# In[Assign files names]
# 獲取當前日期和時間
current_datetime = datetime.now()

# 將當前日期格式化為 "YYYYMMDD" 形式的字串
formatted_date = 20240920


storage= False #看是否需要存檔所有需要的模型, mean, Scaler, 和features



# In[Final model]

results = pd.DataFrame(columns=[ 'AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
results_cmuh = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
results_wk = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])
results_an = pd.DataFrame(columns=['AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])


best_number=28
features = x.columns.tolist()
feature_sel = [features[i] for i in indices]
best_features = feature_sel[best_number:]
print (best_features)
n = len(best_features)

x_train_best = x_train[best_features]
x_test_best = x_test[best_features]
x_cmuh_best = x_cmuh[best_features]
x_wk_best = x_wk[best_features]
x_an_best = x_an[best_features]


# train set
x_train_imp, m_mean = imputer(x_train_best, train=True)
x_train_scale, train_scaler = scaling(x_train_imp, None, train=True)


x_test_imp = imputer(x_test_best, False, m_mean)
x_test_scale, train_scaler = scaling(x_test_imp, train_scaler, train=False)

# ext set 
x_cmuh_imp = imputer(x_cmuh_best, False, m_mean)
x_cmuh_scale, train_scaler = scaling(x_cmuh_imp, train_scaler, train=False)

# WK
x_wk_imp = imputer(x_wk_best, False, m_mean)
x_wk_scale, train_scaler = scaling(x_wk_imp, train_scaler, train=False)

# AN   
x_an_imp = imputer(x_an_best, False, m_mean)
x_an_scale, train_scaler = scaling(x_an_imp, train_scaler, train=False)



y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_cmuh_bin = label_binarize(y_cmuh, classes=[0, 1, 2])
y_wk_bin = label_binarize(y_wk, classes=[0, 1, 2])
y_an_bin = label_binarize(y_an, classes=[0, 1, 2])

 ## Internal ##


model = CatBoostClassifier(bagging_temperature= 1.0,  border_count=56, random_strength=2.93,
                           depth= 6, iterations= 1259, l2_leaf_reg= 3,
                           learning_rate= 0.035,  verbose= 0)

## Internal ##
X_train_resampled, y_train_resampled= x_train_scale, y_train
# model fit

model.fit(X_train_resampled, y_train_resampled)


y_prob = model.predict_proba(x_test_scale)
y_prob_bin = custom_classify(y_prob, thresholds)
y_pred = y_prob_bin.argmax(axis=1)

# 在這裡計算每個類別的AUROC和AUPRC
auroc = roc_auc_score(y_test_bin, y_prob, average=None)
auprc = average_precision_score(y_test_bin, y_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics, cm = get_metrics(y_test, y_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_df = pd.DataFrame(metrics).T
class_metrics_df['AUROC'] = auroc
class_metrics_df['AUPRC'] = auprc
results = pd.concat([results, class_metrics_df])

## External ## cmuh

y_cmuh_prob = model.predict_proba(x_cmuh_scale)
y_cmuh_prob_bin = custom_classify(y_cmuh_prob, thresholds)
y_cmuh_pred = y_cmuh_prob_bin.argmax(axis=1)

# 在這裡計算每個類別的AUROC和AUPRC
auroc_cmuh = roc_auc_score(y_cmuh_bin, y_cmuh_prob, average=None)
auprc_cmuh = average_precision_score(y_cmuh_bin, y_cmuh_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics_cmuh, cm_cmuh = get_metrics(y_cmuh, y_cmuh_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_cmuh = pd.DataFrame(metrics_cmuh).T
class_metrics_cmuh['AUROC'] = auroc_cmuh
class_metrics_cmuh['AUPRC'] = auprc_cmuh
results_cmuh = pd.concat([results_cmuh, class_metrics_cmuh])

## External ## wk

y_wk_prob = model.predict_proba(x_wk_scale)
y_wk_prob_bin = custom_classify(y_wk_prob, thresholds)
y_wk_pred = y_wk_prob_bin.argmax(axis=1)


# 在這裡計算每個類別的AUROC和AUPRC
auroc_wk = roc_auc_score(y_wk_bin, y_wk_prob, average=None)
auprc_wk = average_precision_score(y_wk_bin, y_wk_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics_wk, cm_wk = get_metrics(y_wk, y_wk_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_wk = pd.DataFrame(metrics_wk).T
class_metrics_wk['AUROC'] = auroc_wk
class_metrics_wk['AUPRC'] = auprc_wk
results_wk = pd.concat([results_wk, class_metrics_wk])



## External ## an
y_an_prob = model.predict_proba(x_an_scale)
y_an_prob_bin = custom_classify(y_an_prob, thresholds)
y_an_pred = y_an_prob_bin.argmax(axis=1)


# 在這裡計算每個類別的AUROC和AUPRC
auroc_an = roc_auc_score(y_an_bin, y_an_prob, average=None)
auprc_an = average_precision_score(y_an_bin, y_an_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics_an, cm_an = get_metrics(y_an, y_an_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_an = pd.DataFrame(metrics_an).T
class_metrics_an['AUROC'] = auroc_an
class_metrics_an['AUPRC'] = auprc_an
results_an = pd.concat([results_an, class_metrics_an])


print (f'train set performance= {results}')
print (f'test set performance= {results_cmuh}')
print (f'wk set performance= {results_wk}')
print (f'an set performance= {results_an}')

if storage==True:
    filename_model = rf"{variable_path}\GS_{formatted_date}_{best_number}_model.joblib"
    filename_imputer = rf"{variable_path}\GS_{formatted_date}_{best_number}_imputer.joblib"
    filename_scaler = rf"{variable_path}\GS_{formatted_date}_{best_number}_Scaler.joblib"
    filename_features = rf"{variable_path}\GS_{formatted_date}_{best_number}_best_features.pkl"

    dump(best_features, filename_features)
    dump(model, filename_model)
    dump(m_mean, filename_imputer)  # Assuming m_mean is your imputer
    dump(train_scaler, filename_scaler)

    
    # 將預測結果合併回原本檔案
    data_cmuh['predictions'] = y_cmuh_pred  # 將預測結果作為新列加入
    data_WK['predictions'] = y_wk_pred  # 將預測結果作為新列加入
    data_AN['predictions'] = y_an_pred  # 將預測結果作為新列加入
    
    file_name= rf"{result_path}\GS_predictions_output.xlsx"
    writer = pd.ExcelWriter(file_name, engine='openpyxl')
    
    # 將 DataFrame 寫入不同的 Excel 工作表
    data_cmuh.to_excel(writer, sheet_name='Ext Set Predictions')
    data_WK.to_excel(writer, sheet_name='WK Predictions')
    data_AN.to_excel(writer, sheet_name='AN Predictions')
    
    # 關閉 writer 對象
    writer.close()


# In[Shaple value]

import shap

shap.initjs()

best_feature= ['WBC', 'SD_MALS_LY', 'MN_UMALS_NE', 'SD_UMALS_EO', 'MN_AL2_LY', 'SD_MALS_EO',
                'SD_C_NE', 'MN_C_EO', 'PLR', 'SD_V_EO', 'MN_UMALS_EO', 'MN_LALS_NE', 'SD_LMALS_EO', 'SD_LALS_LY', 'MN_LMALS_EO',
                'SD_UMALS_LY', 'SD_AL2_MO', 'SD_MALS_MO', 'SD_AL2_LY', 'ANC', 'MN_C_NE',
                'SD_C_LY', 'EO_count', 'MN_AL2_NE', 'SD_LMALS_LY', 'MCHC', 'SD_LMALS_MO', 'SD_LALS_EO', 'SD_UMALS_MO',
                'SD_AL2_EO', 'RBC', 'SD_C_EO', 'MCH', 'SD_LALS_NE', 'MCV', 'LY_count', 'MN_V_LY', 'SD_AL2_NE', 'HCT', 'NE_count',
                'SD_V_LY', 'SD_C_MO', 'MN_V_NE', 'SD_UMALS_NE', 'MN_V_EO', 'NLR', 'MN_C_MO', 'MO_count', 'MN_LALS_EO', 'MN_C_LY', 'MN_V_MO',
                'MO', 'RDW', 'SD_V_NE',  'PLT', 'SD_LALS_MO', 'MDW', 'NE']


x_train_best = x_train[best_feature]
x_test_best = x_test[best_feature]

# train set
x_train_imp, m_mean = imputer(x_train_best, train=True)
x_train_scale, train_scaler = scaling(x_train_imp, None, train=True)


x_test_imp = imputer(x_test_best, False, m_mean)
x_test_scale, train_scaler = scaling(x_test_imp, train_scaler, train=False)


model = CatBoostClassifier(bagging_temperature= 1.0,  border_count=56, random_strength=2.93,
                           depth= 6, iterations= 1259, l2_leaf_reg= 3,
                           learning_rate= 0.035,  verbose= 0)

model.fit(x_train_scale, y_train)

target_class=2

explainer = shap.TreeExplainer(model)

#shap_values = explainer.shap_values(x_train_selected)
shap_values = explainer(x_train_scale) # 2 items, 指定其中一個, Catboost 不需要[1]不需要[1]import shap

# 仅获取目标类别的 SHAP 值
shap_values_target_class = shap_values[:,:, target_class]


shap.summary_plot(shap_values_target_class, features=x_train_scale, plot_type="dot", 
                  plot_size= 0.6,max_display= 15, feature_names=x_train_scale.columns)
# 如果shap_values 沒有指定維度, 就會變成multiple class, 變成長條圖




# In[Import data]
    
    
data_origin = pd.read_csv(r'D:\張裕鑫\RESEARCH\CPD\急診預測\20230927\20231014 update 刪掉沒有長出細菌名稱\CMUH_202108_202212_4_hours_gramstain.csv')
data_cmuh = pd.read_csv(r'D:\張裕鑫\RESEARCH\CPD\急診預測\20231003\20231013 修正\CMUH_202301_202308_4_hours_gramstain_2.csv')
data_WK = pd.read_csv (r'D:\張裕鑫\RESEARCH\CPD\急診預測\20231005 gramstain WK\1013 沒有名稱的格蘭氏陽性當作沒長\WK_202301_202308_4_hours_gramstain.csv')

data= data_origin.copy()



best_features_catboost=['SD_UMALS_EO', 'EO', 'MN_V_NE', 'SD_UMALS_MO',
                        'ANC', 'SD_MALS_LY', 'MO_count', 'SD_C_LY', 'MN_LMALS_NE',
                        'EO_count', 'SD_AL2_EO', 'MN_AL2_EO', 'MN_V_EO', 'MN_V_MO',
                        'SD_MALS_NE', 'SD_UMALS_NE', 'MN_LMALS_MO', 'SD_C_NE',
                        'MN_AL2_LY', 'NE', 'MN_LMALS_LY', 'HGB', 'SD_LALS_EO',
                        'SD_AL2_MO', 'SD_LALS_NE', 'MN_AL2_MO', 'MN_LALS_NE', 
                        'MN_LALS_LY', 'RBC', 'MN_UMALS_EO', 'SD_LMALS_MO', 'PLT',
                        'MN_MALS_NE', 'MO', 'SD_V_MO', 'MCHC', 'SD_LALS_LY', 
                        'SD_LALS_MO', 'SD_V_NE', 'SD_V_EO', 'MN_C_EO', 'MN_V_LY',
                        'PDW', 'MDW', 'MN_C_NE', 'NLR', 'BA', 'MN_C_LY', 'MN_C_MO']

data = data.loc[:, best_features_catboost]

'''
# best features
columns_to_keep = [
    'MN_AL2_EO', 'NLR', 'SD_V_EO', 'MN_V_MO', 'SD_AL2_EO', 'RBC', 
    'SD_LALS_MO', 'SD_V_MO', 'MN_V_NE', 'MN_V_EO', 'MO', 'SD_V_NE', 
    'MCHC', 'MN_LMALS_LY', 'EO', 'MDW', 'PDW', 'PLT', 'MN_V_LY', 
    'MN_C_EO', 'BA', 'MN_C_MO', 'MN_C_NE', 'MN_C_LY', 'gram_final'
]
'''

'''
columns_to_keep = [
    'MDW', 'RBC', 'MN_UMALS_NE', 'MN_V_LY', 'MN_LMALS_MO', 'MN_C_NE', 
    'SD_LALS_LY', 'SD_LALS_MO', 'MN_V_MO', 'MN_C_LY', 'MO', 'MN_AL2_LY', 
    'SD_LMALS_MO', 'MN_LALS_NE', 'SD_V_EO', 'gram_final'
]

# 用 .loc[] 來選擇 DataFrame 的子集
data = data.loc[:, columns_to_keep]
'''

'''
#columns_to_remove = ["MN_C_LY", "MN_C_NE",  "MN_C_MO", "BA", "MN_LALS_LY", "MN_MALS_MO", ""]
columns_to_remove = [ "MN_C_LY", "MN_C_NE",    "MN_LALS_LY" , "PDW", "EO", "MN_C_EO"]
# 移除這些欄位
data = data.drop(columns=columns_to_remove)
'''
# In[將特徵和標籤分開]
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 指定x,y
y_cmuh = data_cmuh.iloc[:,-1]
x_cmuh = data_cmuh.iloc[:,:-1]

y_wk = data_WK.iloc[:,-1]
x_wk = data_WK.iloc[:,:-1]

# In
total_start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_seed)

results = pd.DataFrame(columns=['Model', 'AUROC','AUPRC','Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])





y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_scale = scaler.fit_transform(X_train_imp)
X_test_scale = scaler.transform(X_test_imp)

joblib.dump(scaler, f"scaler")

X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_scale, y_train)
        
# model fit

model= LGBMClassifier()
#model= XGBClassifier()
#model= RandomForestClassifier()
#model= LogisticRegression()
#model= CatBoostClassifier(verbose=False)
#model= SVC(gamma='auto',probability=True)

model.fit(X_train_resampled, y_train_resampled)
joblib.dump(model, f"model")

y_pred = model.predict(X_test_scale)
y_prob = model.predict_proba(X_test_scale)
y_prob_bin = label_binarize(y_prob.argmax(axis=1), classes=[0, 1, 2])

# 在這裡計算每個類別的AUROC和AUPRC
auroc = roc_auc_score(y_test_bin, y_prob, average=None)
auprc = average_precision_score(y_test_bin, y_prob, average=None)

# 將每個類別的AUROC和AUPRC存儲在列表中
metrics, cm_model = get_metrics(y_test, y_pred)

# 将class_metrics字典转换为DataFrame
class_metrics_df = pd.DataFrame(metrics).T
class_metrics_df['AUROC'] = auroc
class_metrics_df['AUPRC'] = auprc
results = pd.concat([results, class_metrics_df])


total_end_time = time.time() # 記錄結束時間
total_elapsed_time = total_end_time - total_start_time  # 計算所花費的時間
print(f"total took {total_elapsed_time:.2f} seconds to run.")
# In[]

# model = joblib.load(f'model')

importances = model.feature_importances_
indices = np.argsort(importances) [-15::][:]
features = X.columns
plt.subplots(figsize=(6,9))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"rf_feature_importance.png", dpi = 300, facecolor='white', transparent = True, bbox_inches = "tight")
plt.show()





