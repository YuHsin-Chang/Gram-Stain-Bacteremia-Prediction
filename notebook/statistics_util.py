import pandas as pd
import os
import numpy as np
from joblib import dump, load
from datetime import datetime
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# In[parameter setting]
    
random_seed= 564
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
data_cmuh= pd.read_csv(fr'{data_path}\{data_date}\CMUH 2023\CMUH_GS_2023.csv')
data_WK = pd.read_csv(fr'{data_path}\{data_date}\WK\WK_GS_2023.csv')
data_AN = pd.read_csv(fr'{data_path}\{data_date}\AN\AN_GS_2023.csv')

# [將特徵和標籤分開]
y = data.iloc[:, -1]
y_cmuh = data_cmuh.iloc[:,-1]
y_wk = data_WK.iloc[:,-1]
y_an = data_AN.iloc[:,-1]


x = data.iloc[:,6:93]
x_cmuh = data_cmuh.iloc[:,6:93]
x_wk = data_WK.iloc[:,6:93]
x_an = data_AN.iloc[:,6:93]

data= pd.concat([x,y],axis=1)
data_cmuh= pd.concat([x_cmuh,y_cmuh],axis=1)
data_wk= pd.concat([x_wk,y_wk],axis=1)
data_an= pd.concat([x_an,y_an],axis=1)


data['source']='CMUH_developing'
data_cmuh['source']='CMUH_validation'
data_wk['source']='WMH'
data_an['source']='ANH'

data_combined= pd.concat([data,data_cmuh,data_wk,data_an], axis=0).reset_index(drop=True)

# In[Statistic] 統計全部的參數

data_list=['CMUH_developing', 'CMUH_validation', 'WMH', 'ANH']
total_results_df=pd.DataFrame()

# Loop through each column to calculate mean, variance and perform t-test
for source in data_list:
    results = []
    for col in data_combined.columns:
        if col == 'source' or col == 'final_label':
            continue
            
        print(source, col)
        
        
        mean_1 = data_combined[x_combine['source'] == source][col].median()
        std_1 = data_combined[x_combine['source'] == source][col].std()

    
        # Append the results for the current column to the results list
        results.append({
            'Parameter': col,
            f'Mean_Group_{source}': f'{mean_1:.1f} ({std_1:.1f})'
        })
    
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    total_results_df= pd.concat([total_results_df,results_df],axis=1)
    total_results_df.to_csv(f'{result_path}\\Statistics_total.csv', index=True, encoding='utf_8_sig')



# In[箱型圖加violin]

variable='SD_V_NE'
y_label_name= variable.replace('_','-')
box_width =0.06
# 設置圖表風格，去掉背景色但保留網格線
sns.set(style="whitegrid")

# 設置圖表大小
plt.figure(figsize=(10, 5), dpi=600)

# 分別繪製 Label==0 和 Label==1 的小提琴圖和箱線圖，並設定樣式

sns.violinplot(x='source', y=variable, hue='final_label', data=data_combined, split=False, inner=None, 
               edgecolor='Black', linewidth=1.5,
               width=0.7, dodge=True, palette={0:'#3d5c88', 1:'#d7e0ed', 2:'black'})


deviate=0.23
# 手動添加箱線圖的部分

# 先找出上下緣
MAX=data_combined[variable].max()
MIN=data_combined[variable].min()

sns.boxplot(x='source', y=variable, hue='final_label', data=data_combined[data_combined['final_label'] == 0],
             width=box_width,  # 更改這裡的寬度
             showfliers=False, dodge=False, notch=False,
             boxprops={'linewidth': 1.2, 'linestyle':'-', 'edgecolor': 'black', 'facecolor':'white'},  # 設定箱線圖的邊框樣式
             whiskerprops={'linewidth': 1.5, 'color':'black'},  # 設定鬍鬚線的寬度
             medianprops={'linewidth': 2, 'color':'black'},   # 設定中位數線條的寬度
             capprops={'linewidth':0 },      # 隱藏末端的橫線
             zorder=1,                       # 確保箱線圖在小提琴圖上層
             positions=[0-deviate,1-deviate,2-deviate,3-deviate]
            )
deviate=0
sns.boxplot(x='source', y=variable, hue='final_label', data=data_combined[data_combined['final_label'] == 1],
             width=box_width,  # 更改這裡的寬度
             showfliers=False, dodge=False, notch=False,
             boxprops={'linewidth': 1.2, 'edgecolor': 'black', 'facecolor':'white'},  # 設定箱線圖的邊框樣式
             whiskerprops={'linewidth': 1.5},  # 設定鬍鬚線的寬度
             medianprops={'linewidth': 2},   # 設定中位數線條的寬度
             capprops={'linewidth': 0},      # 隱藏末端的橫線
             zorder=1,                       # 確保箱線圖在小提琴圖上層
             positions=[0+deviate,1+deviate,2+deviate,3+deviate]
            )

deviate=0.23
sns.boxplot(x='source', y=variable, hue='final_label', data=data_combined[data_combined['final_label'] == 2],
             width=box_width,  # 更改這裡的寬度
             showfliers=False, dodge=False, notch=False,
             boxprops={'linewidth': 1.2, 'edgecolor': 'black', 'facecolor':'white'},  # 設定箱線圖的邊框樣式
             whiskerprops={'linewidth': 1.5},  # 設定鬍鬚線的寬度
             medianprops={'linewidth': 2},   # 設定中位數線條的寬度
             capprops={'linewidth': 0},      # 隱藏末端的橫線
             zorder=1,                       # 確保箱線圖在小提琴圖上層
             positions=[0+deviate,1+deviate,2+deviate,3+deviate]
            )



x_pos = [0,1,2,3]  # x 軸位置，對應於 cohort 的索引位置
y_pos = MAX*1.05  # y 軸位置，可以根據數據的範圍手動調整

for x in x_pos:
    plt.text(x-deviate/2, y_pos, '**', fontsize=16, color='black', ha='center')
    plt.text(x+deviate/2, y_pos, '**', fontsize=16, color='black', ha='center')
    plt.plot([x-deviate, x-deviate, x-0.02, x-0.02], [y_pos, y_pos +(MAX-MIN)*0.02, y_pos +(MAX-MIN)*0.02,y_pos], lw=1.0, color='black')
    plt.plot([x+0.02, x+0.02, x+deviate, x+deviate], [y_pos, y_pos +(MAX-MIN)*0.02, y_pos +(MAX-MIN)*0.02,y_pos], lw=1.0, color='black')
    
    plt.text(x, y_pos+(MAX-MIN)*0.06, '**', fontsize=16, color='black', ha='center')
    plt.plot([x-deviate, x-deviate, x+deviate, x+deviate], [y_pos+(MAX-MIN)*0.06, y_pos+(MAX-MIN)*0.08 , y_pos +(MAX-MIN)*0.08,y_pos+(MAX-MIN)*0.06], lw=0.7, color='black')
    
    
# 調整黑色軸線
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_color('black')

# 自定義圖例，只顯示兩個圖例項目
handles = [plt.Line2D([0], [0], color="#2b415f", lw=5, label="Nonbacteremia"),
           plt.Line2D([0], [0], color="#d7e0ed", lw=5, label="Gram-negative"),
           plt.Line2D([0], [0], color="black", lw=5, label="Gram-positive")]


plt.legend(handles=handles, fontsize=9)

# 設置標題和標籤
plt.title(f'{y_label_name} Distribution by Cohort and Label', fontsize=16)
plt.xlabel('Cohort', fontsize=14)
plt.ylabel(f'{y_label_name}', fontsize=14)

# 設置灰色網格，並將背景設置為白色
plt.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray')

# 調整 y 軸範圍，根據實際數據調整
plt.ylim(MIN-(MAX-MIN)*0.15, MAX+ (MAX-MIN)*0.20)

# 顯示圖表
plt.show()


# [Statistic] 統計全部的參數
import scikit_posthocs as sp
import pandas as pd

data_list=['CMUH_developing', 'CMUH_validation', 'WMH', 'ANH']
total_results_df=pd.DataFrame()

# Loop through each column to calculate mean, variance and perform t-test
for source in data_list:
    # Extract data for Dunn's test
    data = data_combined[[variable, 'final_label']]
    
    # Perform Dunn’s test with Bonferroni correction for multiple comparisons
    dunn_result = sp.posthoc_dunn(data, val_col=variable, group_col='final_label', p_adjust='bonferroni')
    
    # Display the results
    print("Dunn's Test Result with Bonferroni correction:")
    print(dunn_result)


# In[箱型圖]

variable='SD_V_NE'
y_label_name= variable.replace('_','-')
box_width =0.12
# 設置圖表風格，去掉背景色但保留網格線
sns.set(style="whitegrid")

# 設置圖表大小
plt.figure(figsize=(8, 5), dpi=600)

deviate=0.23
# 手動添加箱線圖的部分

# 先找出上下緣
MAX=data_combined[variable].quantile(0.975)
MIN=data_combined[variable].quantile(0.03)

sns.boxplot(x='source', y=variable, hue='final_label', data=data_combined[data_combined['final_label'] == 0],
             width=box_width,  # 更改這裡的寬度
             showfliers=False, dodge=False, notch=False,
             boxprops={'linewidth': 1.2, 'linestyle':'-', 'edgecolor': 'black', 'facecolor':'whitesmoke'},  # 設定箱線圖的邊框樣式
             whiskerprops={'linewidth': 1.5, 'color':'black'},  # 設定鬍鬚線的寬度
             medianprops={'linewidth': 2, 'color':'black'},   # 設定中位數線條的寬度
             capprops={'linewidth':1 , 'color':'black'},      # 隱藏末端的橫線
             zorder=1,                       # 確保箱線圖在小提琴圖上層
             positions=[0-deviate,1-deviate,2-deviate,3-deviate]
            )
deviate=0
sns.boxplot(x='source', y=variable, hue='final_label', data=data_combined[data_combined['final_label'] == 1],
             width=box_width,  # 更改這裡的寬度
             showfliers=False, dodge=False, notch=False,
             boxprops={'linewidth': 1.2, 'edgecolor': 'black', 'facecolor':'cornflowerblue'},  # 設定箱線圖的邊框樣式
             whiskerprops={'linewidth': 1.5, 'color':'black'},  # 設定鬍鬚線的寬度
             medianprops={'linewidth': 2, 'color':'black'},   # 設定中位數線條的寬度
             capprops={'linewidth': 1, 'color':'black'},      # 隱藏末端的橫線
             zorder=1,                       # 確保箱線圖在小提琴圖上層
             positions=[0+deviate,1+deviate,2+deviate,3+deviate]
            )

deviate=0.23
sns.boxplot(x='source', y=variable, hue='final_label', data=data_combined[data_combined['final_label'] == 2],
             width=box_width,  # 更改這裡的寬度
             showfliers=False, dodge=False, notch=False,
             boxprops={'linewidth': 1.2, 'edgecolor': 'black', 'facecolor':'lightcoral'},  # 設定箱線圖的邊框樣式
             whiskerprops={'linewidth': 1.5, 'color':'black'},  # 設定鬍鬚線的寬度
             medianprops={'linewidth': 2, 'color':'black'},   # 設定中位數線條的寬度
             capprops={'linewidth': 1, 'color':'black'},      # 隱藏末端的橫線
             zorder=1,                       # 確保箱線圖在小提琴圖上層
             positions=[0+deviate,1+deviate,2+deviate,3+deviate]
            )



x_pos = [0,1,2,3]  # x 軸位置，對應於 cohort 的索引位置
y_pos = MAX*1.04  # y 軸位置，可以根據數據的範圍手動調整
# 1-2
plt.text(0+deviate/2, y_pos, '**', fontsize=16, color='black', ha='center')
plt.text(1+deviate/2, y_pos, '***', fontsize=16, color='black', ha='center')
plt.text(2+deviate/2, y_pos+0.55, 'ns', fontsize=10, color='black', ha='center')
plt.text(3+deviate/2, y_pos+0.55, 'ns', fontsize=10, color='black', ha='center')

# 0-1
plt.text(0-deviate/2, y_pos, '***', fontsize=16, color='black', ha='center')
plt.text(1-deviate/2, y_pos, '***', fontsize=16, color='black', ha='center')
plt.text(2-deviate/2, y_pos, '***', fontsize=16, color='black', ha='center')
plt.text(3-deviate/2, y_pos, '***', fontsize=16, color='black', ha='center')

# 0-2
plt.text(0, y_pos+(MAX-MIN)*0.1, '***', fontsize=16, color='black', ha='center')
plt.text(1, y_pos+(MAX-MIN)*0.1, '***', fontsize=16, color='black', ha='center')
plt.text(2, y_pos+(MAX-MIN)*0.1, '***', fontsize=16, color='black', ha='center')
plt.text(3, y_pos+(MAX-MIN)*0.1, '***', fontsize=16, color='black', ha='center')

for x in x_pos:
    #plt.text(x-deviate/2, y_pos, '***', fontsize=16, color='black', ha='center')
    #plt.text(x+deviate/2, y_pos, '***', fontsize=16, color='black', ha='center')
    #plt.text(x, y_pos+(MAX-MIN)*0.1, '***', fontsize=16, color='black', ha='center')
    plt.plot([x-deviate, x-deviate, x-0.02, x-0.02], [y_pos, y_pos +(MAX-MIN)*0.02, y_pos +(MAX-MIN)*0.02,y_pos], lw=1.0, color='black')
    plt.plot([x+0.02, x+0.02, x+deviate, x+deviate], [y_pos, y_pos +(MAX-MIN)*0.02, y_pos +(MAX-MIN)*0.02,y_pos], lw=1.0, color='black')    
    plt.plot([x-deviate, x-deviate, x+deviate, x+deviate], [y_pos+(MAX-MIN)*0.1, y_pos+(MAX-MIN)*0.12 , y_pos +(MAX-MIN)*0.12,y_pos+(MAX-MIN)*0.1], lw=0.7, color='black')
    
    
# 調整黑色軸線
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_color('black')

# 自定義圖例，只顯示兩個圖例項目
handles = [plt.Line2D([0], [0], color="whitesmoke", lw=5, label="Nonbacteremia"),
           plt.Line2D([0], [0], color="cornflowerblue", lw=5, label="Gram-negative"),
           plt.Line2D([0], [0], color="lightcoral", lw=5, label="Gram-positive")]


plt.legend(handles=handles, fontsize=9)

# 設置標題和標籤
plt.title(f'{y_label_name} Distribution by Cohort and Label', fontsize=16)
plt.xlabel('Cohort', fontsize=14)
plt.ylabel(f'{y_label_name}', fontsize=14)

# 設置灰色網格，並將背景設置為白色
plt.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray')

# 調整 y 軸範圍，根據實際數據調整
plt.ylim(MIN-(MAX-MIN)*0.5, MAX+ (MAX-MIN)*0.35)
plt.xlim(-0.5, 3.5)

# 顯示圖表
plt.show()


# [Statistic] 統計全部的參數
import scikit_posthocs as sp
import pandas as pd

data_list=['CMUH_developing', 'CMUH_validation', 'WMH', 'ANH']
total_results_df=pd.DataFrame()

# Loop through each column to calculate mean, variance and perform t-test
for source in data_list:
    # Extract data for Dunn's test
    data = data_combined.loc[data_combined['source']==source,[variable, 'final_label']]
    
    # Perform Dunn’s test with Bonferroni correction for multiple comparisons
    dunn_result = sp.posthoc_dunn(data, val_col=variable, group_col='final_label', p_adjust='bonferroni')
    
    # Display the results
    print("Dunn's Test Result with Bonferroni correction:")
    #print(dunn_result)
    print(round(dunn_result,4))
