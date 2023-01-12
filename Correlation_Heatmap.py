## 00. Clear
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import warnings
warnings.filterwarnings(action='ignore')

## 01. Import library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## 02. Loading the data
df = pd.read_excel("Raw_data.xlsx")
X = df.drop(['Adsorbent','H', 'O', 'N', 'Pollutant', 'Wastewater type', 'Adsorption type', 'Final concentration'],axis=1)

## 03. Correlation heatmap
plt.figure(figsize=(18,15))
sns.set(font_scale=1.5)
sns.heatmap(data = X.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='RdYlBu_r',annot_kws={"size": 13})
plt.tight_layout()