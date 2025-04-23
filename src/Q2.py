import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy as sp
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn import linear_model
from tqdm import tqdm
from dataloader import DataLoader
warnings.filterwarnings('ignore')

os.chdir('../src')
dataloader = DataLoader()

dataset = dataloader.load_data()
dataset.columns
dataset['1stFlrSF']
dataset['PricePerArea'] = dataset['SalePrice'] / dataset['LotArea']
# ---------------------------
# 💰 지역별 '평단가' 기반 등급 분류 (5단계)
# ---------------------------
price_per_area_by_neigh = dataset['PricePerArea']
q20 = price_per_area_by_neigh.quantile(0.20)
q40 = price_per_area_by_neigh.quantile(0.40)
q60 = price_per_area_by_neigh.quantile(0.60)
q80 = price_per_area_by_neigh.quantile(0.80)

def classify_price_grade(price):
    if price <= q20:
        return 1
    elif price <= q40:
        return 2
    elif price <= q60:
        return 3
    elif price <= q80:
        return 4
    else:
        return 5

# dataset['PriceGrade'] = dataset['PricePerArea'].apply(classify_price_grade)

#  위험도 평균 열 생성
dataset['Risk_Avg'] = (
    dataset['Risk_RoofMatl'] * 0.30 +
    dataset['Risk_Exterior1st'] * 0.30 +
    dataset['Risk_Exterior2nd'] * 0.10 +
    dataset['Risk_MasVnrType'] * 0.10 +
    dataset['Risk_WoodDeckSF'] * 0.2
)

# 위험도 평균을 5단계로 그룹화
dataset['Risk_Level'] = dataset['Risk_Avg'].round()
dataset['Risk_Level'].value_counts().sort_index()
dataset.groupby('Risk_Level')['PricePerArea'].mean()
# 결측값 제거
dataset = dataset.dropna(subset=['PricePerArea'])

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('PricePerArea ~ C(Risk_Level)',data=dataset).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

# 해당 그림이 0을 기준으로 잘 분포되어있어야함 (잔차의 정규성)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.scatter(model.fittedvalues, model.resid)

# 애는 만족안함
import scipy.stats as sp
W, p = sp.shapiro(model.resid)
print(f'검정통계량: {W:.3f}, 유의확률: {p:.3f}')

# 애는 아님
from scipy.stats import probplot
plt.figure(figsize=(6, 6))
probplot(model.resid, dist="norm", plot=plt)


# 등분산성 검정 (만족)
from scipy.stats import bartlett
from scipy.stats import kruskal
groups = [1, 2, 3, 4, 5]
grouped_residuals = [model.resid[dataset['Risk_Level'] == group] for group in groups]
test_statistic, p_value = bartlett(*grouped_residuals)
print(f"검정통계량: {test_statistic}, p-value: {p_value}")

# 변수명을 dataset으로 바꾸고 Kruskal-Wallis 검정 다시 실행

# 그룹 나누기
grouped = [group['PricePerArea'].values for name, group in dataset.groupby('Risk_Level')]

# Kruskal-Wallis 검정
kruskal_stat, kruskal_p = kruskal(*grouped)

# 결과 반환
kruskal_result = {
    "검정통계량 (H)": kruskal_stat,
    "p-value": kruskal_p,
    "결론": "✔️ 그룹 간 차이가 유의함 (p < 0.05)" if kruskal_p < 0.05 else "❌ 유의한 차이 없음 (p ≥ 0.05)"
}

kruskal_result


# 비모수 사후검정
import scikit_posthocs as sp
posthoc = sp.posthoc_dunn(dataset, val_col='PricePerArea', group_col='Risk_Level', p_adjust='bonferroni')
posthoc

