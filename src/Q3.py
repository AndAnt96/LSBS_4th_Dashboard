import os
from dataloader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import pickle
import joblib

import warnings
warnings.filterwarnings('ignore')

# os.chdir('./src')

# 피해액 책정 모델 구축
def price_prediction_models(ames):
    """라쏘 및 릿지 회귀를 활용한 피해액 예측 모델 구축"""
    target = 'SalePrice'
    y = ames[target]
    
    ig_cols = ['PID','SalePrice','GeoRefNo','Prop_Addr','Latitude','Longitude','Date','Risk_RoofMatl','Risk_Exterior1st','Risk_Exterior2nd','Risk_MasVnrType','Risk_WoodDeckSF']
    ames = ames.drop(columns=ig_cols)
    
    X = ames 
    
    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_columns = X_train.select_dtypes('number').columns.tolist()
    cat_columns = X_train.select_dtypes('object').columns.tolist()

    cat_preprocess = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    num_preprocess = make_pipeline(SimpleImputer(strategy="mean"), 
                                StandardScaler())

    preprocessor = ColumnTransformer(
    [("num", num_preprocess, num_columns),
    ("cat", cat_preprocess, cat_columns)]
    )

    # # Elastic 모델
    elastic_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(max_iter=10000))
    ])
    
    # 알파값 설정
    elastic_params = {
        'regressor__alpha':np.arange(0.1, 1, 0.1),
        'regressor__l1_ratio': np.linspace(0,1,5)
    }
    
    # K-폴드 교차 검증
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lasso 모델 튜닝
    elastic_grid = GridSearchCV(
        elastic_pipeline, 
        elastic_params, 
        cv=kfold, 
        scoring='neg_mean_squared_error'
    )
    
    elastic_grid.fit(X_train, y_train)
    best_elastic = elastic_grid.best_estimator_
    y_pred = best_elastic.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return best_elastic, y_pred, mse, rmse, r2

def estimate_bill_price(ames, predicted):
    
    
    return None



def main(ames):
    print("===== 화재 발생 시 예상 피해액 모델링 =====\n")
    
    dataloader = DataLoader()
    ames = dataloader.load_data()
    
    
    # 피해액 책정 모델 구축
    best_model, predictions, mse, rmse, r2  = price_prediction_models(ames)
    best_model.get_params()
    joblib.dump(best_model, 'best_elastic_model.pkl') 
    
    print(f"ELASTIC")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.4f}")
    
if __name__ == "__main__":

    dataloader = DataLoader()
    dataset = dataloader.load_data()
    
    # risk_columns = [c for c in dataset.columns if c.split('_')[0] == 'Risk']
    # risk_columns
    
    main(dataset)
