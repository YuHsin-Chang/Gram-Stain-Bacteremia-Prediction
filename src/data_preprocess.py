# %%
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score , average_precision_score, confusion_matrix
from tqdm import tqdm
# %%

def ED_list_process(df_path) :
    new_columns = {
        '病歷號': 'ID',
        '急診號': 'ED_ID',
        '身分證': 'Personal_ID',
        '就診日': 'EDdate',
        '時間': 'EDtime',
        '生日': 'Birthday',
        '年齡': 'Age',
        '離院日': 'ED_DIS_date',
        '時間.1': 'ED_DIS_time',
        '診斷碼': 'ICD',
        '診斷碼1': 'ICD_1',
        '診斷碼2': 'ICD_2',
        '診斷碼3': 'ICD_3',
        '診斷碼4': 'ICD_4',
        '診斷碼5': 'ICD_5',
        '轉歸名稱': 'Disposition',
        '血壓PE': 'SBP',
        '血壓BP': 'DBP',
        '脈搏': 'HR',
        '體溫': 'BT',
        '體重': 'Body weight',
        '身高': 'Body Height',
        '依據': 'Triage_depend',
        'E': 'GCS_E',
        'M': 'GCS_M',
        'V': 'GCS_V',
        '呼吸': 'RR',
        '身上管路': 'Tubes',
        '住院日': 'ADM_date',
        '出院日': 'ADM_DIS_date',
        '住院天數': 'ADM_days',
        '住院科別': 'ADM_specialty',
        '住院床別': 'ADM_bed_type',
        '檢傷': 'Triage_level',
        '性別': 'Gender',
        '轉入': 'Transferred',
        '轉入院所': 'Transfer_from',
        '急診停留時間(小時)': 'ED_LOS',
        '放射線費':'Fee_radiation',
        '檢驗檢查費':'Fee_examination',
        '其他': 'Fee_other',
        '總金額': 'Fee_total',
        '檢傷主訴': 'C/C', # Chief complaint
        '科別代碼': 'ED_specialty_code',
        '科別名稱': 'ED_specialty',
        '看診日期': 'Assessdate',
        '時間.2': 'Assesstime',
        '候診時間(分鐘)': 'Waiting_time',
        '到院方式':'Ambulatory_way',
        '死亡診斷書': 'Death_certificate',
        '病歷present illness': 'Present_illness',
        'P.H.': 'Past_history',
        '入院區': 'ED_ADM_zone',  #ADM admission
        '離院區': 'ED_DIS_zone',   #DIS discharge
        '轉出': 'Transfer_to',
        '三日返診': '3D_return',
        '前次返診時間差': 'Return_timespan',
        '中央靜脈': 'CVC',
        '尿管': 'FOLEY',
        '急診抗生素藥品(明細)': 'ED_ABs', #antibiotics
        '一日返診': '1D_return',
        '前次離院時間': 'Last_DIStime'
        }
    
    
    df = pd.read_csv(df_path, encoding='utf-8-sig')
    df.rename(columns=new_columns, inplace=True)
    
    return df

def CD4_xlsx_process(df_path) :
    new_columns = {
        'ChartNo': 'ID',
        'ReportTime': 'ReportTime_CD4'
        }
    
    
    df = pd.read_excel(df_path, skiprows=2, sheet_name=0, engine='openpyxl')
    df.rename(columns=new_columns, inplace=True)
    
    return df


def combine_datetime_MK(date, time):
    try:
        date = pd.to_numeric(date)
        time = pd.to_numeric(time)
    except (ValueError, TypeError):
        return None

    try: 
        if date==0:
            year=1800
            month=1
            day=1
            hour=0
            minute=0
            
        else:
            year = int(date // 10000) +1911
            month = int(date % 10000 / 100)
            day = int(date % 100)
            hour = int(time / 100)
            minute = int(time % 100)

        
        return datetime(year, month, day, hour, minute)
    
    except ValueError as e:
        print(f"Error with date {date} and time {time}: {e}")
    return None

def combine_datetime(date, time):
    # 檢查是否為數值型別
    try:
        date = pd.to_numeric(date)
        time = pd.to_numeric(time)
    except (ValueError, TypeError):
        return None
    
    # 檢查是否為空值
    if pd.isna(time) or pd.isna(date):
        return None
    
    # 提取日期與時間
    year = int(date / 10000)
    month = int((date % 10000) / 100)
    day = int(date % 100)
    hour = int(time / 100)
    minute = int(time % 100)

    # 檢查月份是否合理
    if month > 12 or month < 1 or day < 1 or day > 31:
        return None

    try:
        # 回傳組合好的日期時間
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None

def combine_datetime_sec(date, time):
    # 檢查是否為數值型別
    try:
        date = pd.to_numeric(date)
        time = pd.to_numeric(time)
    except (ValueError, TypeError):
        return None
    
    # 檢查是否為空值
    if pd.isna(time) or pd.isna(date):
        return None
    
    # 提取日期與時間
    year = int(date / 10000)
    month = int((date % 10000) / 100)
    day = int(date % 100)
    hour = int(time / 10000)
    minute = int(time % 10000/100)
    sec = int(time % 100)
    # 檢查月份是否合理
    if month > 12 or month < 1 or day < 1 or day > 31:
        return None

    try:
        # 回傳組合好的日期時間
        return datetime(year, month, day, hour, minute, sec)
    except ValueError:
        return None

def check_column(value):
    if  pd.isna(value) or '100' in value:  #有一些兒科醫師的代碼不是D開頭, 而是100開頭
        return False
    
    return value[0] != 'D'


def convert_to_datetime(value):
    value= str(value)
    
    try:
        # 分離日期和時間部分
        if ' ' in value:
            date_str, time_str = value.split()
        else:
            date_str, time_str = value, None

        # 解析日期部分
        year = int(date_str[0:3]) + 1911
        month = int(date_str[3:5])
        day = int(date_str[5:7])

        # 解析時間部分
        if time_str:
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            dt = datetime(year, month, day, hour, minute)
            return dt
        else:
            dt = datetime(year, month, day)
            return dt
        
        
    except Exception as e:           
        return pd.NaT  # 如果解析失败，返回 None 或其他适当的值


def performance (y_test, predictions, y_prob):
    conf_matrix = confusion_matrix(y_test, predictions)
    #print ({'confusion_matrix':conf_matrix})
    true_positive= conf_matrix[1,1]
    true_negative= conf_matrix[0,0]
    false_positive= conf_matrix[0,1]
    false_negative= conf_matrix[1,0]
    total= true_positive+true_negative+false_positive+false_negative
    precision= true_positive/(true_positive+false_positive)  #positive predictive value
    recall= true_positive/(true_positive+false_negative)   # sensitivity
    specificity= true_negative/(true_negative+false_positive)
    negative_predictive_value= true_negative/(true_negative+false_negative)
    accuracy= (true_positive+true_negative)/total
    F1_score=(2*precision*recall)/(precision+recall)
    AUROC= roc_auc_score (y_test, y_prob)
    AUPRC= average_precision_score (y_test, y_prob)
    return conf_matrix, [{'AUROC':AUROC, 'AUPRC':AUPRC, 'accuracy':accuracy ,'F1_score':F1_score, 'recall':recall, 'specificity':specificity\
                   , 'precision':precision, 'NPV':negative_predictive_value}]
# %% 標定function
# 汙染菌定義

"""
- H12有其中報告有 Type : 4的菌且間隔24小時內中有出現同樣的菌，且不為"Staphylococcus coagulase-negative" | "Coagulase negative Staphylococcus spp.(CNS）" -> 1: 有長菌
- H12有其中報告有 Type : 4的菌且24小時內Type4的菌都不重複 ->2: 排除
"""
def check_contamination(Sample_df, ori_label, contamination_list):
    Base_Time = Sample_df.CBC_Time.iloc[0]
    contamination_dict = {}
    for st ,v in zip(Sample_df.Sample_Time, Sample_df.Value) :
        bac_names = find_bac_name(v) 
        for bac_name in bac_names :
            if bac_name in contamination_list :
                if bac_name not in contamination_dict :
                    contamination_dict[bac_name] = [st]
                else :
                    contamination_dict[bac_name].append(st)
    new_label = ori_label
    for k, v in contamination_dict.items() :
        if len(v) == 1 :
            continue
        v = sorted([abs(_v - Base_Time).seconds/60/60 for _v in v])
        valid1 = np.abs(v)<12
        valid2 = list(np.diff(v)<24)
        valid3 = np.array(valid2+[valid2[-1]])
        valid2 = np.array([valid2[0]]+valid2)
        if np.any(valid1 & valid2) or np.any(valid1 & valid3) :
            if k in ["Staphylococcus coagulase-negative" ,"Coagulase negative Staphylococcus spp.(CNS）"] :
                new_label = 2
            else :
                new_label = 1
                break
        
    return new_label

def check_contamination_wholeED(Sample_df, ori_label, contamination_list):
    contamination_dict = {}
    for st ,v in zip(Sample_df.Sample_Time, Sample_df.Value) :
        bac_names = find_bac_name(v) 
        for bac_name in bac_names :
            if bac_name in contamination_list : # 配對汙染菌採檢的時間
                if bac_name not in contamination_dict :  
                    contamination_dict[bac_name] = [st]
                else :
                    contamination_dict[bac_name].append(st)
    new_label = ori_label
    for bacteria, sample_times in contamination_dict.items() :
        if len(sample_times) == 1 :
            continue
        
        # 只要算出有任一距離<24小時即可代表是陽性
        sorted_times = sorted(sample_times)
        time_diffs = np.diff(sorted_times)
        time_diffs_hours = np.array([td.total_seconds() / 3600 for td in time_diffs])
        
        # 判斷是否有任兩次採檢時間差小於24小時
        if any(time_diffs_hours < 24):
            if bacteria in ["Staphylococcus coagulase-negative", "Coagulase negative Staphylococcus spp.(CNS）"]:
                new_label = 2  # 排除
            else:
                new_label = 1  # 陽性
                break
        
    return new_label

def str_post_process(the_segment, where = 'first', the_key = None) :
    the_segment = the_segment.split(';')
    the_segment = [s.lstrip().rstrip() for s in the_segment] #去頭去尾空白
    the_segment = [s for s in the_segment if s != '']
    # if the_key is not None :
    #     the_segment = [s for s in the_segment if ~ the_key in s]
    if not the_segment :
        return None
    if where == 'first' :
        if the_key is not None :
            return the_segment[0] if not (the_key in the_segment[0]) else None
        return the_segment[0] if the_segment[0] not in ["S :", "R :", "I :"] else the_segment[0]
    if where == 'end' :
        if the_key is not None :
            return the_segment[-1] if not (the_key in the_segment[-1]) else None
        return the_segment[-1] if the_segment[-1] not in ["S :", "R :", "I :"] else the_segment[-2]

def segment_find(the_segment, the_first = False, the_end = False) :
    if the_segment == '' :
        return None

    # % 規則 1: 查找 "Antimicrobial" 上一行的內容
    key = 'Antimicrobial'
    if key in the_segment :
        return str_post_process(the_segment.split(key)[0], 'end')
    # % 規則 2: 若出現 "Candida"，"Cryptococcus" 那一行就是細菌名稱
    key = 'Candida'
    if key in the_segment :
        lines = the_segment.split(';')
        line_to_process = next((line for line in lines if key in line), None)
        return str_post_process(line_to_process,'first') #the_segment.split(key)[1], 'first') => 這樣會把key 切掉

    key = 'Cryptococcus'
    if key in the_segment :
        lines = the_segment.split(';')
        line_to_process = next((line for line in lines if key in line), None)
        return str_post_process(line_to_process,'first') #(the_segment.split(key)[1], 'first') => 這樣會把key 切掉

    # %  規則 3: "嗜氧報告" 和 "厭氧報告" 下一行是細菌名稱，除非出現 "Blood culture no growth for X days"
    # "染色結果為 Gram's stain:Gram Negative Bacilli," % 例外1 
    key = '嗜氧報告:'
    if key in the_segment :
        r = the_segment.split(key)[1]
        key1 = 'Gram Negative Bacilli'
        if key1 in r :
            return key1
        key1 = 'Blood culture no growth for'
        if key1 not in r :
            return str_post_process(r, 'first')

    key = '厭氧報告:'
    if key in the_segment :
        r = the_segment.split(key)[1]

        key1 = 'Gram Negative Bacilli'
        if key1 in r :
            return key1
        key1 = 'Blood culture no growth for'
        if key1 not in r :
            return str_post_process(r, 'first')

    # % 規則 4: 查找 "Gram's stain" 前一句，如果不包含 "S: ", "R: ", "I: "，則為細菌名稱
    if (not the_end) and str_post_process(the_segment, 'end', 'S :') and str_post_process(the_segment, 'end', 'R :') and str_post_process(the_segment, 'end', 'I :')  :
        return str_post_process(the_segment, 'end')
    # % 規則 5: 查找 "Penicillin(P)" 前一句為細菌名稱
    key = 'Penicillin(P)'
    if key in the_segment :
        return str_post_process(the_segment.split(key)[0], 'end')

    # % 規則 6: 若上面的方法都沒有找到, 則將 "Gram's stain:" 後的行作為細菌名稱
    if not the_first :
        return str_post_process(the_segment, 'first')
    return the_segment
    
def find_bac_name(the_report) :
    the_report = the_report.replace('\r','').replace('_x000D_','').replace('\n',';').replace('：',':').replace('）',')')
    the_report = the_report.split("Gram's stain:") if "Gram's stain:" in the_report else ['']
    if len(the_report) ==1 :
        return ['']
    bac_names = [segment_find(the_segment, the_first=idx==0, the_end=idx==len(the_report)-1) for idx, the_segment in enumerate(the_report) ]
    bac_names = [bac_name for bac_name in bac_names if bac_name != '染色結果為']
    bac_names = [bac_name for bac_name in bac_names if bac_name is not None]
    bac_names = bac_names[:len(the_report)-1]
    return bac_names


# 定義入ICU和出ICU的邏輯
def assign_ICU_times(group):
    # 初始化列
    group['入ICU時間'] = None
    group['出ICU時間'] = None
    #print (group.index)
    # 遍歷每一行，按照條件更新入ICU和出ICU時間
    indices = group.index.to_list()
    #print (group)
    for i in indices:
        print (i)
        if 'CU' in group.loc[i, '床位']:
            group.loc[i, '入ICU時間'] = group.loc[i, 'BED_change_Time']
            # 如果有下一筆記錄，將其 BED_change_Time 作為出ICU時間
            if i + 1 <= group.index.to_list()[-1]:
                group.loc[i, '出ICU時間'] = group.loc[i + 1, 'BED_change_Time']
            else:
                # 如果是最後一筆，沒有下一個 BED_change_Time，出ICU時間設為出院日
                group.loc[i, '出ICU時間'] = group.loc[i, '出院日']
    return group


def calculate_mews(row):
    score = 0

    # 呼吸率 RR
    if row['RR'] <= 8:
        score += 3
    elif 9 <= row['RR'] <= 14:
        score += 0
    elif 15 <= row['RR'] <= 20:
        score += 1
    elif 21 <= row['RR'] <= 29:
        score += 2
    elif row['RR'] >= 30:
        score += 3

    # 心跳 HR
    if row['HR'] <= 40:
        score += 2
    elif 41 <= row['HR'] <= 50:
        score += 1
    elif 51 <= row['HR'] <= 100:
        score += 0
    elif 101 <= row['HR'] <= 110:
        score += 1
    elif 111 <= row['HR'] <= 129:
        score += 2
    elif row['HR'] >= 130:
        score += 3

    # 收縮壓 SBP
    if row['SBP'] <= 70:
        score += 3
    elif 71 <= row['SBP'] <= 80:
        score += 2
    elif 81 <= row['SBP'] <= 100:
        score += 1
    elif 101 <= row['SBP'] <= 199:
        score += 0
    elif row['SBP'] >= 200:
        score += 2

    # 體溫 BT
    if row['BT'] < 35:
        score += 2
    elif 35.0 <= row['BT'] < 38.5:
        score += 0
    elif row['BT'] >= 38.5:
        score += 2

    # 意識狀態 GCS_M（可粗略對應）
    if row['GCS_M'] < 4:  # GCS Motor <6 表示不是完全清醒，可視為 VPU
        score += 3
    elif row['GCS_M'] == 4: 
        score += 2
    elif row['GCS_M'] == 5: 
        score += 1
    elif row['GCS_M'] == 6: 
        score += 0
        
    return score



def calculate_SIRS(row):
    score = 0
    if row['BT'] > 38 or row['BT'] < 36:
        score += 1
    if row['HR'] > 90:
        score += 1
    if row['RR'] > 20:
        score += 1
    if row['WBC'] > 12 or row['WBC'] < 4:  # 假設 WBC 單位是 *10^9/L
        score += 1
    return score

