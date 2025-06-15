# %%
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import  StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import label_binarize
from datetime import datetime
from config import cache_dir
from catboost import CatBoostClassifier

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

def get_metrics(y_true, y_pred, y_prob, average=None):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)  # [class0, class1, class2]
    FN = cm.sum(axis=1) - np.diag(cm)  # [class0, class1, class2]
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

   # Calculate class-wise performance
    class_metrics = {}
    num_classes = cm.shape[0]

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    auroc = roc_auc_score(y_true_bin, y_prob, average=None)
    auprc = average_precision_score(y_true_bin, y_prob, average=None)

    for i in range(num_classes):
        class_accuracy = (TP[i] + TN[i]) / (TP[i] + FP[i] + FN[i] + TN[i])
        class_sensitivity = TP[i] / (TP[i] + FN[i])
        class_specificity = TN[i] / (TN[i] + FP[i])
        class_ppv = TP[i] / (TP[i] + FP[i])
        class_npv = TN[i] / (TN[i] + FN[i])
        class_f1 = f1_score(y_true == i, y_pred == i)
       
        class_metrics[f'Class {i}'] = {
            'AUROC': auroc[i],
            'AUPRC': auprc[i],   
            'Accuracy': class_accuracy,
            'F1 Score': class_f1,
            'Sensitivity': class_sensitivity,
            'Specificity': class_specificity,
            'PPV': class_ppv,
            'NPV': class_npv
        }

    class_metrics_df = pd.DataFrame(class_metrics).T
        
    return class_metrics_df, cm



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



def imputer(df, numerical_impute_method='mean', train=True, num_impute_values=None):
    """
    Impute missing values in a DataFrame consisting only of numerical columns.

    Parameters:
    - df: pandas DataFrame (numerical columns only)
    - numerical_impute_method: str, one of ['mean', 'median', 'zero', 'knn', 'mice']
    - train: bool, whether this is training phase (fit) or validation (transform only)
    - num_impute_values: statistics or fitted imputer object from training phase

    Returns:
    - df_impute: DataFrame after imputation
    - num_impute_values: fitted imputer or statistics for reuse in validation
    """
    
    df_impute = df.copy()

    if train:
        if numerical_impute_method == 'mean':
            num_impute_values = df.mean()
            df_impute = df.fillna(num_impute_values)

        elif numerical_impute_method == 'median':
            num_impute_values = df.median()
            df_impute = df.fillna(num_impute_values)

        elif numerical_impute_method == 'zero':
            num_impute_values = 0
            df_impute = df.fillna(0)

        elif numerical_impute_method == 'knn':
            knn_imputer = KNNImputer()
            df_impute = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns, index=df.index)
            num_impute_values = knn_imputer

        elif numerical_impute_method == 'mice':
            mice_imputer = IterativeImputer(max_iter=30)
            df_impute = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns, index=df.index)
            num_impute_values = mice_imputer

        else:
            raise ValueError("Unsupported impute method. Choose from ['mean', 'median', 'zero', 'knn', 'mice'].")

        return df_impute, num_impute_values

    else:
        # Validation or test phase
        if isinstance(num_impute_values, (KNNImputer, IterativeImputer)):
            df_impute = pd.DataFrame(num_impute_values.transform(df), columns=df.columns, index=df.index)
        else:
            df_impute = df.fillna(num_impute_values)

        return df_impute


def find_classwise_thresholds_by_youden(y_true, y_prob, num_classes=3):
    """
    Computes the best threshold for each class using the Youden index.

    Parameters:
    - y_true: 1D array-like of true multiclass labels
    - y_prob: 2D array-like of predicted probabilities for each class (shape: [n_samples, n_classes])
    - num_classes: int, number of classes

    Returns:
    - thresholds: list of best thresholds for each class
    """
    thresholds = []

    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]

        fpr, tpr, thresh = roc_curve(y_true_binary, y_prob_class)
        youden_index = tpr - fpr
        best_thresh = thresh[np.argmax(youden_index)]
        thresholds.append(best_thresh)

    return thresholds




def backward_feature_selection(x_train, y_train, x_val, y_val, min_features=5, 
                                impute_method='mean', random_seed=42,
                                checkpoint_dir=f'{cache_dir}', 
                                checkpoint_name='backward_selection',
                                resume_from_checkpoint=True):
    """
    Backward feature selection function with checkpoint saving and resuming support
    
    Parameters:
    - x_train, y_train: Training data
    - x_val, y_val: Validation data
    - min_features: Minimum number of features to retain
    - impute_method: Imputation method
    - random_seed: Random seed
    - checkpoint_dir: Checkpoint save directory
    - checkpoint_name: Checkpoint filename prefix
    - resume_from_checkpoint: Whether to attempt resuming from checkpoint
    """

    # Checkpoint file paths
    checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name}_checkpoint.pkl')
    progress_path = os.path.join(checkpoint_dir, f'{checkpoint_name}_progress.json')
    
    # Initialize variables
    features_selected = x_train.columns.tolist()
    performance_history = []
    iteration = 0
    
    # Multi-class binarization
    classes = np.unique(y_train)
    y_train_bin = label_binarize(y_train, classes=classes)
    y_val_bin = label_binarize(y_val, classes=classes)
    
    # Attempt to resume from checkpoint
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            features_selected = checkpoint_data['features_selected']
            performance_history = checkpoint_data['performance_history']
            iteration = checkpoint_data['iteration']
            
            print(f"Resuming from checkpoint: iteration {iteration}, remaining features: {len(features_selected)}")
            print(f"Previously removed features: {[item['removed'] for item in performance_history]}")
            
        except Exception as e:
            print(f"Cannot load checkpoint: {e}")
            print("Starting from scratch...")
    
    def save_checkpoint():
        """Save checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_data = {
            'features_selected': features_selected,
            'performance_history': performance_history,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save binary checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Save readable progress log
        progress_data = {
            'current_features_count': len(features_selected),
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'performance_history': performance_history
        }
        
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    # Save initial state
    if iteration == 0:
        save_checkpoint()
    
    try:
        while len(features_selected) > min_features:
            best_macro_auc = -np.inf
            worst_feature = None
            best_class_aucs = None
            best_micro_auc = None
            
            print(f"\n=== Iteration {iteration + 1} ===")
            print(f"Current number of features: {len(features_selected)}")
            
            for feature in tqdm(features_selected, desc=f"Testing removal of features (current: {len(features_selected)})"):
                temp_features = [f for f in features_selected if f != feature]
                
                # Select data
                x_train_sel = x_train[temp_features]
                x_val_sel = x_val[temp_features]
                
                # Preprocessing
                x_train_imp, m_mean = imputer(x_train_sel, impute_method, train=True)
                x_train_scale, train_scaler = scaling(x_train_imp, None, train=True)

                x_val_imp = imputer(x_val_sel, impute_method, False, m_mean)
                x_val_scale, train_scaler = scaling(x_val_imp, train_scaler, train=False)

                # Model training
                
                model = CatBoostClassifier(verbose=False, class_weights={0: 1, 1: 1, 2:1}, random_seed=random_seed)
                model.fit(x_train_scale, y_train)
                
                # Probability output (n_samples, n_classes)
                y_prob = model.predict_proba(x_val_scale)
                
                # Individual class AUROC
                class_aucs = {}
                for i, cls in enumerate(classes):
                    class_aucs[f'class_{cls}_AUROC'] = roc_auc_score(y_val_bin[:, i], y_prob[:, i])
                
                # Macro and micro AUROC
                macro_auc = roc_auc_score(y_val_bin, y_prob, average='macro', multi_class='ovr')
                micro_auc = roc_auc_score(y_val_bin, y_prob, average='micro', multi_class='ovr')
                
                if macro_auc > best_macro_auc:
                    best_macro_auc = macro_auc
                    best_micro_auc = micro_auc
                    worst_feature = feature
                    best_class_aucs = class_aucs
            
            # Remove the worst feature
            features_selected.remove(worst_feature)
            
            # Record results
            performance_history.append({
                'iteration': iteration,
                'n_features': len(features_selected),
                'removed': worst_feature,
                **best_class_aucs,
                'macro_AUROC': best_macro_auc,
                'micro_AUROC': best_micro_auc,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Removed feature: {worst_feature}")
            print(f"Remaining features: {len(features_selected)}")
            print(f"Macro AUROC: {best_macro_auc:.4f}")

            iteration += 1
            # Save checkpoint after each iteration
            save_checkpoint()
            
    except KeyboardInterrupt:
        print("\nInterruption detected, saving checkpoint...")
        save_checkpoint()
        print("Checkpoint saved, can resume later.")
        raise
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Saving checkpoint...")
        save_checkpoint()
        raise
    
 
    print("Feature selection completed. Checkpoint files kept.")

    
    return features_selected, pd.DataFrame(performance_history)


def list_checkpoints(checkpoint_dir=f'{cache_dir}'):
    """List all available checkpoints"""
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint directory does not exist")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_progress.json')]
    
    if not checkpoint_files:
        print("No checkpoint files found")
        return
    
    print("Available checkpoints:")
    for file in checkpoint_files:
        file_path = os.path.join(checkpoint_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            print(f"  {file}: iteration {data['iteration']}, features {data['current_features_count']}, time {data['timestamp']}")
        except Exception as e:
            print(f"  {file}: cannot read - {e}")


def clean_checkpoints(checkpoint_dir=f'{cache_dir}', checkpoint_name=None, confirm=False):
    """Clean checkpoint files"""
    if not confirm:
        print("You must set confirm=True to delete checkpoints.")
        return
    
    if not os.path.exists(checkpoint_dir):
        return
    
    if checkpoint_name:
        # Clean specific checkpoint
        files_to_remove = [
            f'{checkpoint_name}_checkpoint.pkl',
            f'{checkpoint_name}_progress.json'
        ]
    else:
        # Clean all checkpoints
        files_to_remove = [f for f in os.listdir(checkpoint_dir) 
                          if f.endswith('_checkpoint.pkl') or f.endswith('_progress.json')]
    
    for file in files_to_remove:
        file_path = os.path.join(checkpoint_dir, file)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")



#%% =============================Plot=============================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Plot multiclass ROC curve
def plot_multiclass_roc(y_true, y_prob, class_names):
    """
    Plot the ROC curve for each class in a multiclass classification task.

    Parameters:
    - y_true: Ground truth labels (shape: [n_samples])
    - y_prob: Predicted probability scores (shape: [n_samples, n_classes])
    - class_names: List of class names corresponding to the columns in y_prob
    """
    plt.figure(figsize=(6, 6), dpi=600)

    for i, class_name in enumerate(class_names):
        # Binarize true labels for one-vs-rest ROC curve
        y_true_bin = (y_true == i).astype(int)

        # Compute False Positive Rate (FPR) and True Positive Rate (TPR)
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])

        # Compute AUROC (Area Under the ROC Curve)
        auc = roc_auc_score(y_true_bin, y_prob[:, i])

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{class_name} (AUROC={auc:.3f})')

    # Plot diagonal line representing random chance
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='by chance')

    # Set plot labels and formatting
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('AUROC - Development Cohort', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot multiclass Precision-Recall (PR) curve
def plot_multiclass_prc(y_true, y_prob, class_names):
    """
    Plot the Precision-Recall curve for each class in a multiclass classification task.

    Parameters:
    - y_true: Ground truth labels (shape: [n_samples])
    - y_prob: Predicted probability scores (shape: [n_samples, n_classes])
    - class_names: List of class names corresponding to the columns in y_prob
    """
    plt.figure(figsize=(6, 6), dpi=600)

    for i, class_name in enumerate(class_names):
        # Binarize true labels for one-vs-rest PR curve
        y_true_bin = (y_true == i).astype(int)

        # Compute precision-recall values
        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:, i])

        # Compute average precision (AUPRC)
        ap = average_precision_score(y_true_bin, y_prob[:, i])

        # Plot the precision-recall curve
        plt.plot(recall, precision, label=f'{class_name} (AUPRC={ap:.3f})')

    # Set plot labels and formatting
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('AUPRC - Development Cohort')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_combined_calibration_curve(y_bins_list, y_probs_list, labels, num_classes=3):
    """
    Plot combined calibration curves for multi-class classification.

    Parameters:
    - y_bins_list: List of binarized ground truth arrays (one-hot encoded).
    - y_probs_list: List of predicted probability arrays for each class.
    - labels: List of dataset names (used for legend).
    - num_classes: Number of classes (default is 3).
    """

    # Titles for each class-specific calibration plot
    class_titles = {
        0: 'Calibration Plot of Nonbacteremia Prediction',
        1: 'Calibration Plot of Gram Negative Bacteremia Prediction',
        2: 'Calibration Plot of Gram Positive Bacteremia Prediction'
    }

    # Loop through each class and create a plot
    for class_idx in range(num_classes):
        plt.figure(figsize=(10, 8), dpi=600)

        # Plot calibration curves for each dataset
        for i in range(len(y_bins_list)):
            y_bin = y_bins_list[i]         # Ground truth (one-hot) for dataset i
            y_prob = y_probs_list[i]       # Predicted probabilities for dataset i

            # Compute Brier score for this class and dataset
            brier_score = brier_score_loss(y_bin[:, class_idx], y_prob[:, class_idx])

            # Compute calibration curve: fraction of positives vs. mean predicted values
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_bin[:, class_idx], y_prob[:, class_idx], n_bins=10
            )

            # Plot calibration curve with Brier score in label
            plt.plot(
                mean_predicted_value, fraction_of_positives, "s-",
                label=f"{labels[i]} - (Brier score: {brier_score:.3f})"
            )

        # Plot the diagonal line => perfect calibration
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        # Title and axis settings
        title_str = class_titles.get(class_idx, f"Calibration Plot - Class {class_idx}")
        plt.title(title_str, fontsize=18)
        plt.xlabel("Mean predicted value", fontsize=14)
        plt.ylabel("Fraction of positives", fontsize=14)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='best', fontsize=15)
        plt.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray')
        plt.show()
