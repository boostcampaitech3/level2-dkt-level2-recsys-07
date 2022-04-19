import pandas as pd
import numpy as np

train_df=pd.read_csv("/opt/ml/input/data/train_data.csv")
outlier_userID=[5887, 6283, 6382, 6764, 7029, 7166, 5498, 5820, 6760, 6988, 7171, 7186, 481]
print('원래 길이: ', len(train_df))
for outlier in outlier_userID:
    drop_idx=train_df[train_df['userID'] == outlier].index
    train_df.drop(inplace=True, axis=0, index=drop_idx)
print('제거 후 길이: ', len(train_df))
train_df.to_csv('/opt/ml/input/data/train_data_outlier.csv')
print('Outlier 제거된 train_data_outlier.csv 생성 완료')