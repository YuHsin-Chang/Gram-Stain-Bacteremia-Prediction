import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# In[function]

# 修改函數來處理多個類別
def plot_calibration_curve_and_brier_multiclass(y_true_bin, y_prob, dataset_name, num_classes=3):
    plt.figure(figsize=(16, 6))  # 設定更大的畫布來顯示多個類別的圖
    
    for i in range(num_classes):
        # 針對每個類別計算校準曲線
        prob_class = y_prob[:, i]  # 取得每個類別的預測機率
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_bin[:, i], prob_class, n_bins=8)
        
        # 計算 Brier score
        brier_score = brier_score_loss(y_true_bin[:, i], prob_class)
        
        # 繪製每個類別的校準圖
        plt.subplot(1, num_classes, i+1)  # 每個類別一個子圖
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'Class {i} (Brier score: {brier_score:.3f})')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.title(f'Class {i} ({dataset_name})')
        plt.legend()
    
    plt.suptitle(f'Calibration plot ({dataset_name}) for {num_classes} classes')
    plt.show()
    
    
# In[]

plot_calibration_curve_and_brier_multiclass(y_test_bin, y_prob, "Test Set", num_classes=3)
plot_calibration_curve_and_brier_multiclass(y_cmuh_bin, y_cmuh_prob, "CMUH Set", num_classes=3)
plot_calibration_curve_and_brier_multiclass(y_wk_bin, y_wk_prob, "WK Set", num_classes=3)
plot_calibration_curve_and_brier_multiclass(y_an_bin, y_an_prob, "AN Set", num_classes=3)



# In[]
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def plot_combined_calibration_curve(y_bins_list, y_probs_list, labels, num_classes=3):
    class_titles = {
       0: 'Calibration Plot of Non-Bacteremia Prediction',
       1: 'Calibration Plot of Gram Negative Bacteremia Prediction',
       2: 'Calibration Plot of Gram Positive Bacteremia Prediction'
   }   
    
    
    
    
    for class_idx in range(num_classes):
        plt.figure(figsize=(10, 8),  dpi=600)
        for i in range(len(y_bins_list)):
            y_bin = y_bins_list[i]
            y_prob = y_probs_list[i]
            brier_score = brier_score_loss(y_bin[:, class_idx], y_prob[:, class_idx])
            
            # Calculate the calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(y_bin[:, class_idx], y_prob[:, class_idx], n_bins=10)
            
            # Plot the calibration curve
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{labels[i]} - (Brier score: {brier_score:.3f})")
    
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        title_str = class_titles.get(class_idx, f"Calibration Plot - Class {class_idx}")

        
        plt.title(title_str, fontsize=18)
        plt.xlabel("Mean predicted value", fontsize=14)
        plt.ylabel("Fraction of positives", fontsize=14)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='best', fontsize=15)
        plt.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray')

        plt.show()




# 使用之前定義的二元化數據和預測機率
y_bins = [y_test_bin, y_cmuh_bin, y_wk_bin, y_an_bin]
y_probs = [y_prob, y_cmuh_prob, y_wk_prob, y_an_prob]
labels = ["CMUH developing", "CMUH validation", "WMH validation", "ANH validation"]

plot_combined_calibration_curve(y_bins, y_probs, labels, num_classes=3)
