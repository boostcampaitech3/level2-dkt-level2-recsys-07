import pandas as pd

# !important data path
ORIGINAL_TRAIN_DATA = "/opt/ml/input/data/train_data.csv"
SPLITED_TRAIN_DATA = "/opt/ml/input/data/split_train_data.csv"
SPLITED_TEST_DATA = "/opt/ml/input/data/split_test_data.csv"

# 원본 train_data.csv
original_train_df = pd.read_csv(ORIGINAL_TRAIN_DATA)
# 원본 dataframe 재정렬 : userID로 먼저 정렬하고, 같은 유저간에는 시간순으로 정렬
original_train_df = original_train_df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
# 전체 사용자의 수 파악
n_user = original_train_df["userID"].nunique()

# 전체 유저 리스트
user_list = sorted(original_train_df["userID"].unique())
print (f"전체 사용자의 수 : {n_user}")

# train과 test에 사용될 사용자의 수 파악
train_n_user = int(n_user * 0.9)
test_n_user = n_user - train_n_user
print (f"train에 사용될 사용자의 수 : {train_n_user}\ntest에 사용될 사용자의 수 : {test_n_user}")


# 0 ~ 6027 번째 user는 train에 사용
# 6028 ~ 6697 번째 user는 test에 사용
test_start_user = user_list[train_n_user]
test_start_index = min(original_train_df[original_train_df["userID"] == test_start_user].index)

splited_train_df = original_train_df.iloc[:test_start_index]
splited_test_df = original_train_df.iloc[test_start_index:]

splited_train_df.to_csv(SPLITED_TRAIN_DATA, index=False)
splited_test_df.to_csv(SPLITED_TEST_DATA, index=False)