# Paper title and data availability
This repository contains codes for the paper entitled " Machine-learning-based prediction and optimization of emerging contaminants' adsorption capacity on biochar materials" authored by Zeeshan Haider Jaffari, Heewon Jeong, Jaegwan Shin, Jinwoo Kwak, Changgil Son, Yong-Gu Lee, Sangwon Kim, Kangmin Chon, Kyung Hwa Cho, also known as the WEIL group at UNIST, Korea. File name "Raw_data" contains the data set of 3,757 data points used in this study to build this model. Besides, The published article can be found on

# Introduction
Biochar materials have recently received considerable recognition as eco-friendly and cost-effective adsorbents capable of effectively removing hazardous emerging contaminants (e.g., pharmaceuticals, herbicides, and fungicides) to aquatic organisms and human health accumulated in aquatic ecosystems. This study accurately predicts the adsorption capacity of ECs toward biochar materials in aqueous solutions using ten tree-based machine learning (ML) models with a large dataset (3,757 data points). The dataset includes 24 input variables, such as pyrolysis conditions for biochar production (3 features), biochar characteristics (3 features), biochar compositions (6 features), and adsorption experimental conditions (12 features). The applied ML models were evaluated with statistical indicators, and the optimal model was selected. Subsequently, the feature importance of the best-performed model was analyzed by the shapley additive explanations (SHAP). Finally, the partial dependence plot (PDP) was utilized as a post-processing technique to investigate the optimized experimental conditions.

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
The evaluation and comparison of the ML model performances showed that the CatBoost model outperformed all other models with the highest test coefficient of determination (0.9433) and lowest mean absolute error (4.95). Furthermore, the SHAP values indicated that the adsorption experimental conditions provided the highest impact on the model prediction for adsorption capacity (41%), followed by the adsorbent composition (35%), adsorbent characterization (20%), and synthesis conditions (3%). The optimized experimental conditions predicted by the modeling were an N/C ratio of 0.017, BET surface area of 1040 m2/g, the content of C(%) contents of 82.1%, pore volume of 0.46 cm3/g, initial ECs concentration of 100 mg/L, type of pollutant (CAR), adsorption type (Single) and adsorption contact time (720 min). 

# Correspondance
If you feel any dificulties in executing these codes, please contact us through email on jaffarizh@hotmail.com or gua01114@gmail.com. Thank you
