# Gram Stain Bacteremia Prediction via Hematological Data and Machine Learning

This repository contains the source code and documentation for a machine learning project designed to predict **bacteremia subtypes**—including non-bacteremia, gram-negative, and gram-positive bacteremia—based on hematological parameters such as complete blood count (CBC), cell population data (CPD), age, and gender.

**Note**: This project is intended for academic demonstration purposes. It does not include actual patient data and is not directly executable without access to the original dataset.

Due to hospital data confidentiality agreements, we are unable to share original data. However, we have included **precomputed model outputs** in the `cache/` directory, which contain predicted probabilities and corresponding ground-truth labels. These are sufficient to reproduce DeLong’s test, calibration plots, and AUROC/AUPRC plots. Notebooks are available in the `notebook/` folder.

---

## 📁 Folder Structure
```
├── cache/             # Placeholder for model outputs  
├── data/              # Placeholder for training and validation data (not included)  
├── notebook/          # Notebooks for experiments and evaluation  
├── src/               # Source code (configuration and utility functions)  
├── .gitignore         # Files and folders to be ignored by Git  
├── LICENSE            # Project license file  
├── README.md          # Project documentation  
└── requirements.txt   # Required Python packages
```


## 📃 File Descriptions

1. `notebook/model_compare.ipynb`:  
   - Model evaluation across different feature sets  
   - Imputation method comparison  
   - Comparison of different class weights
    
2. `notebook/train_tuning.ipynb`:  
   - Feature selection workflow  
   - Hyperparameter tuning  
   - Final model retraining  
   - Calibration plot and AUROC/AUPRC plots 

3. `src/compare_auc_delong_xu.py`:  
   Python implementation of DeLong’s test for ROC AUC comparison  

4. `src/config.py`:  
   Central configuration of paths  

5. `src/model_utils.py`:  
   Utilities for imputation, scaling, feature scaling, metric evaluation, and plotting  

---

## 🧪 Statistical Test for AUC Comparison

We used the Python implementation of DeLong's test for comparing ROC AUCs, adapted from [yandexdataschool/roc_comparison](https://github.com/yandexdataschool/roc_comparison), based on the method described by X. Sun and W. Xu (2014). The function `delong_roc_test` was used for pairwise comparison of models across validation folds.

---

## 📖 Citation

If you use this codebase in your research, please cite or acknowledge the repository:

> Chang Y-H. *Gram-Stain-Bacteremia-Prediction* GitHub, 2025. https://github.com/YuHsin-Chang/Gram-Stain-Bacteremia-Prediction

---

## 🪪 License

This project is licensed under the Apache License. See the [LICENSE](./LICENSE) file for details.

---

## ✉️ Contact

**Author**: Dr. Yuhsin Chang  
**Email**: *jaller251@gmail.com*