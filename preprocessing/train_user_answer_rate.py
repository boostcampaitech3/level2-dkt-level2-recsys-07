# 유저별 정답률이 0.1이하거나 0.95이상인 유저 제거
import pandas as pd
import numpy as np
from tqdm import tqdm

train_df=pd.read_csv("/opt/ml/input/data/train_data.csv")
outlier_userID=[5887, 6283, 6382, 6764, 7029, 7166, 5498, 5820, 6760, 6988, 7171, 7186, 481]

print('원래 길이: ', len(train_df))
for outlier in tqdm(outlier_userID):
    drop_idx=train_df[train_df['userID'] == outlier].index
    train_df.drop(inplace=True, axis=0, index=drop_idx)
print('제거 후 길이: ', len(train_df))

train_df.to_csv('/opt/ml/input/data/train_data_user_answer_rate.csv')
print('유저별 정답률이 너무 크거나 작은 유저가 제거된 train_data_outlier.csv 생성 완료')