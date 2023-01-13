# Paper title and data availability
This repository contains codes for the paper entitled " Machine-learning-based prediction and optimization of emerging contaminants' adsorption capacity on biochar materials" authored by Zeeshan Haider Jaffari, Heewon Jeong, Jaegwan Shin, Jinwoo Kwak, Changgil Son, Yong-Gu Lee, Sangwon Kim, Kangmin Chon, Kyung Hwa Cho, also known as the WEIL group at UNIST, Korea. File name "Raw_data" contains the data set of 3,757 data points used in this study to build this model. Besides, The published article can be found on

# Introduction

# Files in this Repository
∙ Raw_data.xlsx:
Dataset applied in this study (3,757 data points)
∙ Correlation_Heatmap.py:
Correlations between input variables were investigated using heatmap technique.
∙ *_model.py:
Files named “_model.py” were used to build a model utilizing 10 tree-based machine learning ML) algorithms. Applied ML algorithms can be identified by abbreviations (i.e., BA, CB, DT…) at the beginning of the file name. All the required detail processes, including data preprocessing, optimizing hyperparameters, and evaluating the trained model, are included in the code files.
In the data preprocessing process, this study assigned numerical attributes to four categorical input variables, such as adsorbent, pollutant, wastewater type, and adsorption type, with a one-hot encoder. Additionally, each of the numerical input features was standardized by z-value. In the optimizing hyperparameters process, the Bayesian optimization technique was employed. Statistical parameters, including mean absolute error (MAE) and coefficient of determination (R^2), were applied as performance metrics for the model evaluation.
∙ SHAP_CB_model.py:
This code applied SHAP to investigate feature importance and contribution factors in the highest-performing CatBoost model.
∙ PDP_categorical.py, PDP_non_categorical.py:
These two codes are post-processing tools for applying the partial dependence plot (PDP). The two codes are utilized for categorical and numeric data, respectively.

# Findings

# Correspondance
If you feel any dificulties in executing these codes, please contact us through email on jaffarizh@hotmail.com or gua01114@gmail.com. Thank you
