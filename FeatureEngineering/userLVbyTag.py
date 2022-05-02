import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

def percentile(s):
    return round(np.sum(s) / len(s)*100, 2)

def avgLV(data):
    tags = data['KnowledgeTag'].unique()
    LV = []
    
    for tid in tags:
        LV.append(data[data['KnowledgeTag']==tid]['userLVbyTag'].iloc[0])
    return round(np.mean(LV))

warnings.filterwarnings(action='ignore') # CopyWarning 출력 무시 

# 0. 데이터 로드
print(f'[DEBUG]] Loading Data ...')
df = pd.read_csv('/opt/ml/input/data/FE_total_data.csv')
train = df[df['answerCode']>=0]
print(f'[INFO] Total Data: {len(df)}')
print(f'[INFO] Feature Engineering Data (only answerCode >= 0): {len(train)}')
# 1. 태그별 정답률 & 유저별 태그 정답률
print(f'[DEBUG] Calculating correct tagRatio and userBytagRatio ...')
train['tagRatio'] = train.groupby('KnowledgeTag').answerCode.transform(percentile)
train['userBytagRatio'] = train.groupby(['userID', 'KnowledgeTag']).answerCode.transform(percentile)
# 2.유저별 정답률
print(f'[DEBUG] Calculating correct userRatio ...')
train['userRatio'] = train.groupby('userID').answerCode.transform(percentile)
print(f'[INFO] Done!!\n')

feature_by_test = train.drop(['assessmentItemID', 'Timestamp', 'testId', 'answerCode'], axis=1)


# Tag LV
tag_mean_ratio = round(feature_by_test.groupby('KnowledgeTag').mean()['tagRatio'].mean(), 2)
diff =  feature_by_test['tagRatio'] - tag_mean_ratio
pre_std = int(diff.min()//10)*10
last_std = int(diff.max()//10+1)*10
LV = (last_std - pre_std)//10

print('[INFO] Labeling for Tag Level...')
print(f'[INFO] tag mean ratio: {tag_mean_ratio}%')
print(f'[INFO] diff min({round(diff.min(), 2)}%), diff max({round(diff.max(), 2)}%)\n')

for std in tqdm(range(pre_std+10, last_std+1, 10)):
    print(f'[DEBUG] {pre_std:3d}% < diff <= {std:3d}%  |  Level: {LV}')
    idx = (pre_std < diff) & (diff <= std)
    feature_by_test.loc[idx, 'tagLV'] = LV
    pre_std = std
    LV -= 1

print()
print('[INFO] Done!!')
print(f'[INFO] Check all Tag Level: {sorted(feature_by_test["tagLV"].unique())}\n')

# User LV by Tag
diff =  feature_by_test['userBytagRatio'] - feature_by_test['tagRatio']
pre_std = int(diff.min()//10)*10
last_std = int(diff.max()//10+1)*10
LV = 1

print('[INFO] Labeling for User Level by Tag...')
print(f'[INFO] diff min({round(diff.min(), 2)}%), diff max({round(diff.max(), 2)}%)\n')

for std in tqdm(range(pre_std+10, last_std+1, 10)):
    print(f'[DEBUG] {pre_std:3d}% < diff <= {std:3d}%  |  Level: {LV}')
    idx = (pre_std < diff) & (diff <= std)
    feature_by_test.loc[idx, 'userLV'] = LV
    pre_std = std
    LV += 1

print()
print('[INFO] Done!!')
print(f'[INFO] Check all User Level: {sorted(feature_by_test["userLV"].unique())}\n')

print('[DEBUG] Labeling for Total User Level (tagLV x userLV) ...')
feature_by_test['userLVbyTag'] = feature_by_test['tagLV']*feature_by_test['userLV']
print('[INFO] Done!!')
print(f'[INFO] Num of User Level: {len(feature_by_test["userLVbyTag"].unique())}')
print(f'[INFO] Check all User Level: {sorted(feature_by_test["userLVbyTag"].unique())}\n')

warnings.filterwarnings(action='default')

print(f'[DEBUG] Labeling for User Average Level ...')
user_grouby = feature_by_test.groupby('userID').apply(avgLV)
for uid in tqdm(user_grouby.index):
    feature_by_test.loc[feature_by_test['userID']==uid, 'userAvgLV'] = user_grouby.iloc[uid]
levels = feature_by_test['userAvgLV'].unique()
print(f'[INFO] Done!!')
print(f'[INFO] Num of User AVG Levels: {len(levels)}')
print(f'[INFO] Level(min): {min(levels)}, Level(max): {max(levels)}')
print(f'[INFO] Check all User AVG Level: {sorted(levels)}\n')

print(f'[DEBUG] Merge with Original Dataset ...')
idx = feature_by_test.index
columns = ['tagLV', 'userAvgLV']
for column in tqdm(columns):
    df.loc[idx, column] = feature_by_test[column]
df.rename(columns={'userAvgLV':'userLVbyTag'}, inplace=True)

print()
print(f'[DEBUG] Feature Engineering to Inference data ...')
test = df[df['answerCode']<0]
for idx in tqdm(test.index):
    uid = test.loc[idx, 'userID']
    tagid = test.loc[idx, 'KnowledgeTag']
    df.loc[idx, 'userLVbyTag'] = df[df['userID']==uid]['userLVbyTag'].iloc[0]
    df.loc[idx, 'tagLV'] = df[df['KnowledgeTag']==tagid]['tagLV'].iloc[0]
    
print()
print(f'[DEBUG] Saving ...')
df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
tagLV_df = df.drop('userLVbyTag', axis=1)
userLV_df = df.drop('tagLV', axis=1)
tagLV_df.to_csv('/opt/ml/input/data/FE/tagLV.csv', index=False)
userLV_df.to_csv('/opt/ml/input/data/FE/userLVbyTag.csv', index=False)
print('[INFO] Done!!')
print(f'[INFO] Check your "/opt/ml/input/data/FE/tagLV.csv"')
print(f'[INFO] Check your "/opt/ml/input/data/FE/userLVbyTag.csv"')