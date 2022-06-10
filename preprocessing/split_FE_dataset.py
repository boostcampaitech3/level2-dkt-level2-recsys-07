import pandas as pd

# 기본 경로 설정
TRAIN_DATA_PATH = "/opt/ml/input/data/train_data.csv"
TEST_DATA_PATH = "/opt/ml/input/data/test_data.csv"
SPLITED_PATH = "/opt/ml/input/data/FE_total_data.csv"

data1 = pd.read_csv(TRAIN_DATA_PATH)
data2 = pd.read_csv(TEST_DATA_PATH)

data1['dataset'] = 1 # train dataset code
data2['dataset'] = 2 # test dataset code

data = pd.concat([data1, data2])
data = data.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)

train_user = data1['userID'].unique()
test_user = data2['userID'].unique()

print(f'Train file 사용자의 수: {len(train_user)}')
print(f'Test file 사용자의 수: {len(test_user)}')
print(f'Feature Engineering Train 전체 사용자의 수: {len(train_user)+len(test_user)}')

data.to_csv(SPLITED_PATH, index=False)