import pandas as pd

# In[setting]

version_date= 20240920
data_date='20240208 CLSI'
data_path= r'D:\張裕鑫\RESEARCH\CPD\急診預測\格蘭氏染色'
result_path=rf'D:\python_code\GS\result\performance\{version_date}'


# In[Import data]   
# 讀取 Excel 檔案，將第一列作為索引
result_cmuh_origin = pd.read_excel(fr'{result_path}\GS_predictions_output.xlsx', sheet_name='Ext Set Predictions', index_col=0)
result_WK_origin = pd.read_excel(fr'{result_path}\GS_predictions_output.xlsx', sheet_name='WK Predictions', index_col=0)
result_AN_origin = pd.read_excel(fr'{result_path}\GS_predictions_output.xlsx', sheet_name='AN Predictions', index_col=0)

# In[]
# 移除 final_label == 0 的數據, 不需要處理
result_cmuh = result_cmuh_origin[result_cmuh_origin['final_label'] != 0].copy()
result_WK = result_WK_origin[result_WK_origin['final_label'] != 0].copy()
result_AN = result_AN_origin[result_AN_origin['final_label'] != 0].copy()


# In[找出最大的bloodcx_duration_36_i]
# CMUH
columns_cmuh = result_cmuh.columns 
bloodcx_columns = [col for col in columns_cmuh if col.startswith('bloodcx_duration_36_')]
i_values = [int(col.split('_')[-1]) for col in bloodcx_columns]
max_i_cmuh = max(i_values)

# WK
columns_WK = result_WK.columns 
bloodcx_columns = [col for col in columns_WK if col.startswith('bloodcx_duration_36_')]
i_values = [int(col.split('_')[-1]) for col in bloodcx_columns]
max_i_WK = max(i_values)

# AN
columns_AN = result_AN.columns 
bloodcx_columns = [col for col in columns_AN if col.startswith('bloodcx_duration_36_')]
i_values = [int(col.split('_')[-1]) for col in bloodcx_columns]
max_i_AN = max(i_values)


# In[]
def time_to_hours(time_str):
    # 處理負數時間
    negative = 1
    if time_str.startswith('-'):
        negative = -1
        time_str = time_str[1:]
    
    # 將時間字串拆分為時、分、秒
    h, m, s = map(int, time_str.split(':'))
    
    # 計算總小時數
    total_hours = negative * (h + m / 60 + s / 3600)
    return total_hours



# In[]
 # 設定12小時作為閾值
twelve_hours = 12
# 初始化一個空的列表來存放細菌數據，這次每個 index 對應原本的行
result_cmuh.loc[:,'bacteria_list'] = None  # 先初始化一個新列用來存放細菌資料

# 遍歷從 1 到 max_i_cmuh 的所有列
for i in range(1, max_i_cmuh + 1):
    # 構造列名
    duration_col = f'bloodcx_duration_36_{i}'
    bac_1_col = f'bloodcx_36_{i}_bac_1'
    bac_2_col = f'bloodcx_36_{i}_bac_2'
    
    # 確保對應的列存在
    if duration_col in result_cmuh.columns and bac_1_col in result_cmuh.columns and bac_2_col in result_cmuh.columns:
        # 將 'bloodcx_duration_36_x' 列轉換為小時
        result_cmuh.loc[:,'duration_in_hours'] = result_cmuh[duration_col].apply(time_to_hours)
        
        # 篩選出絕對時間小於 12 小時的行
        filtered_df = result_cmuh[abs(result_cmuh['duration_in_hours']) < twelve_hours]

        # 將對應的細菌列加入原始行
        for index, row in filtered_df.iterrows():
            print (i,index)
            bac_1 = row[bac_1_col]
            bac_2 = row[bac_2_col]
            current_list = result_cmuh.at[index, 'bacteria_list']
            # 如果該行已經有細菌列表，則追加細菌資料
            
            if isinstance(current_list, list):
                if any(pd.notna(item) for item in current_list):
                    current_list.extend([bac_1, bac_2])
                    result_cmuh.at[index, 'bacteria_list'] = current_list
                else:
                    result_cmuh.at[index, 'bacteria_list'] = [current_list, bac_1, bac_2]
                    print ('why')
            else:
                result_cmuh.at[index, 'bacteria_list'] = [bac_1, bac_2]

        
result_cmuh.loc[:,'bacteria_list'] = result_cmuh['bacteria_list'].apply(lambda x: list(set(x)) if isinstance(x, list) else x)       

# In[WK]


result_WK.loc[:, 'bacteria_list'] = None  # 先初始化一個新列用來存放細菌資料

# 遍歷從 1 到 max_i_cmuh 的所有列
for i in range(1, max_i_WK + 1):
    # 構造列名
    duration_col = f'bloodcx_duration_36_{i}'
    bac_1_col = f'bloodcx_36_{i}_bac_1'
    bac_2_col = f'bloodcx_36_{i}_bac_2'
    
    # 確保對應的列存在
    if duration_col in result_WK.columns and bac_1_col in result_WK.columns and bac_2_col in result_WK.columns:
        # 將 'bloodcx_duration_36_x' 列轉換為小時
        result_WK.loc[:, 'duration_in_hours'] = result_WK[duration_col].apply(time_to_hours)
        
        # 篩選出絕對時間小於 12 小時的行
        filtered_df = result_WK[abs(result_WK['duration_in_hours']) < twelve_hours]

        # 將對應的細菌列加入原始行
        for index, row in filtered_df.iterrows():
            print (i,index)
            bac_1 = row[bac_1_col]
            bac_2 = row[bac_2_col]
            current_list = result_WK.at[index, 'bacteria_list']
            # 如果該行已經有細菌列表，則追加細菌資料
            
            if isinstance(current_list, list):
                if any(pd.notna(item) for item in current_list):
                    current_list.extend([bac_1, bac_2])
                    result_WK.at[index, 'bacteria_list'] = current_list
                else:
                    result_WK.at[index, 'bacteria_list'] = [current_list, bac_1, bac_2]
                    print ('why')
            else:
                result_WK.at[index, 'bacteria_list'] = [bac_1, bac_2]

        
result_WK.loc[:, 'bacteria_list'] = result_WK['bacteria_list'].apply(lambda x: list(set(x)) if isinstance(x, list) else x)

# In[AN]


result_AN.loc[:, 'bacteria_list'] = None  # 先初始化一個新列用來存放細菌資料

# 遍歷從 1 到 max_i_cmuh 的所有列
for i in range(1, max_i_AN + 1):
    # 構造列名
    duration_col = f'bloodcx_duration_36_{i}'
    bac_1_col = f'bloodcx_36_{i}_bac_1'
    bac_2_col = f'bloodcx_36_{i}_bac_2'
    
    # 確保對應的列存在
    if duration_col in result_AN.columns and bac_1_col in result_AN.columns and bac_2_col in result_AN.columns:
        # 將 'bloodcx_duration_36_x' 列轉換為小時
        result_AN.loc[:, 'duration_in_hours'] = result_AN[duration_col].apply(time_to_hours)
        
        # 篩選出絕對時間小於 12 小時的行
        filtered_df = result_AN[abs(result_AN['duration_in_hours']) < twelve_hours]

        # 將對應的細菌列加入原始行
        for index, row in filtered_df.iterrows():
            print (i,index)
            bac_1 = row[bac_1_col]
            bac_2 = row[bac_2_col]
            current_list = result_AN.at[index, 'bacteria_list']
            # 如果該行已經有細菌列表，則追加細菌資料
            
            if isinstance(current_list, list):
                if any(pd.notna(item) for item in current_list):
                    current_list.extend([bac_1, bac_2])
                    result_AN.at[index, 'bacteria_list'] = current_list
                else:
                    result_AN.at[index, 'bacteria_list'] = [current_list, bac_1, bac_2]
                    print ('why')
            else:
                result_AN.at[index, 'bacteria_list'] = [bac_1, bac_2]

        
result_AN.loc[:, 'bacteria_list'] = result_AN['bacteria_list'].apply(lambda x: list(set(x)) if isinstance(x, list) else x)


# In[統計最常出現的細菌]

# 分別取出 final_label == 1 和 final_label == 0 的數據
result_cmuh_label_1 = result_cmuh[result_cmuh['final_label'] == 1]
result_cmuh_label_2 = result_cmuh[result_cmuh['final_label'] == 2]


# 展開 'bacteria_list' 中的細菌項，將每個列表中的元素分開成獨立的行
bacteria_flat_1 = result_cmuh_label_1['bacteria_list'].explode()
bacteria_counts_cmuh_1 = bacteria_flat_1.value_counts()
# 將展開的細菌進行 one-hot encoding
bacteria_one_hot_1 = pd.get_dummies(bacteria_flat_1)

# 將 one-hot encoding 結果重新與原始數據合併
# 這裡 groupby 原來的索引，並使用 max() 來合併每個細菌對應的行
bacteria_one_hot_grouped_1 = bacteria_one_hot_1.groupby(bacteria_one_hot_1.index).max()

# 將 one-hot encoding 結果與原始 DataFrame 合併
result_CMUH_with_one_hot_1 = pd.concat([result_cmuh_label_1, bacteria_one_hot_grouped_1], axis=1)


# 展開 'bacteria_list' 中的細菌項，將每個列表中的元素分開成獨立的行
bacteria_flat_2 = result_cmuh_label_2['bacteria_list'].explode()
bacteria_counts_cmuh_2 = bacteria_flat_2.value_counts()

# 將展開的細菌進行 one-hot encoding
bacteria_one_hot_2= pd.get_dummies(bacteria_flat_2)

# 將 one-hot encoding 結果重新與原始數據合併
# 這裡 groupby 原來的索引，並使用 max() 來合併每個細菌對應的行
bacteria_one_hot_grouped_2 = bacteria_one_hot_2.groupby(bacteria_one_hot_2.index).max()

# 將 one-hot encoding 結果與原始 DataFrame 合併
result_CMUH_with_one_hot_2 = pd.concat([result_cmuh_label_2, bacteria_one_hot_grouped_2], axis=1)


# In[]

# 1. 排除 "no bac"，並選擇出現次數最多的 10 種細菌
top_10_bacteria_1 = bacteria_counts_cmuh_1[bacteria_counts_cmuh_1.index != 'no bac'].nlargest(11)

# 2. 初始化一個字典來儲存每種細菌的準確率
bacteria_accuracy = {}

# 3. 針對每個細菌，分別計算準確率
for bacteria in top_10_bacteria_1.index:
    # 篩選出包含該細菌的樣本
    samples_with_bacteria = result_cmuh[result_cmuh['bacteria_list'].apply(lambda x: bacteria in x if isinstance(x, list) else False)]
    
    # 計算準確率（final_label 與 predictions 相等的比例）
    accuracy = (samples_with_bacteria['final_label'] == samples_with_bacteria['predictions']).mean()
    
    # 將結果儲存在字典中
    bacteria_accuracy[bacteria] = accuracy

for bacteria, acc in bacteria_accuracy.items():
    print(f"{bacteria} 的準確率: {acc:.2f}")
    
 # In[]   
# 1. 排除 "no bac"，並選擇出現次數最多的 10 種細菌
top_10_bacteria_2 = bacteria_counts_cmuh_2[bacteria_counts_cmuh_2.index != 'no bac'].nlargest(11)

# 2. 初始化一個字典來儲存每種細菌的準確率
bacteria_accuracy = {}

# 3. 針對每個細菌，分別計算準確率
for bacteria in top_10_bacteria_2.index:
    # 篩選出包含該細菌的樣本
    samples_with_bacteria = result_cmuh[result_cmuh['bacteria_list'].apply(lambda x: bacteria in x if isinstance(x, list) else False)]
    
    # 計算準確率（final_label 與 predictions 相等的比例）
    accuracy = (samples_with_bacteria['final_label'] == samples_with_bacteria['predictions']).mean()
    
    # 將結果儲存在字典中
    bacteria_accuracy[bacteria] = accuracy

for bacteria, acc in bacteria_accuracy.items():
    print(f"{bacteria} 的準確率: {acc:.2f}")
    
    

# In[]
bacterias_list_WK = result_WK['bacteria_list'].explode()
bacteria_counts_WK = bacterias_list_WK.value_counts()

bacterias_list_AN = result_AN['bacteria_list'].explode()
bacteria_counts_AN = bacterias_list_AN.value_counts()





# In[]


result_WK_label_1 = result_WK[result_WK['final_label'] == 1]
result_WK_label_2 = result_WK[result_WK['final_label'] == 2]

result_AN_label_1 = result_AN[result_AN['final_label'] == 1]
result_AN_label_2 = result_AN[result_AN['final_label'] == 2]
