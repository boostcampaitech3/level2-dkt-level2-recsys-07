import pandas as pd
import numpy as np
import os
import random

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings(action="ignore")

data_dir = "/opt/ml/input/data/"  # 경로는 상황에 맞춰서 수정해주세요!
csv_file_path = os.path.join(data_dir, "all_feature_data.csv")  # 데이터는 대회홈페이지에서 받아주세요 :)
df = pd.read_csv(csv_file_path)

df.groupby("userID")["answerCode"].transform(lambda x: x.cumsum())

df.groupby("userID")["answerCode"].transform(lambda x: type(x))

df.groupby("userID")["answerCode"].transform(lambda x: x.shift(1))

df["user_correct_answer"] = df.groupby("userID")["answerCode"].transform(
    lambda x: x.cumsum().shift(1)
)

df["user_total_answer"] = df.groupby("userID")["answerCode"].cumcount()

df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]

df.groupby(["testId"])["answerCode"].agg(["mean", "sum"])

df.loc[:]

def feature_engineering(df):

    # 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["user_correct_answer"] = df.groupby("userID")["answerCode"].transform(
        lambda x: x.cumsum().shift(1)
    )
    df["user_total_answer"] = df.groupby("userID")["answerCode"].cumcount()
    df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(["testId"])["answerCode"].agg(["mean", "sum"])
    correct_t.columns = ["test_mean", "test_sum"]
    correct_k = df.groupby(["KnowledgeTag"])["answerCode"].agg(["mean", "sum"])
    correct_k.columns = ["tag_mean", "tag_sum"]

    df = pd.merge(df, correct_t, on=["testId"], how="left")
    df = pd.merge(df, correct_k, on=["KnowledgeTag"], how="left")

    # 카테고리형 feature
    categories = ["assessmentItemID", "testId"]

    for category in categories:
        df[category] = df[category].astype("category")

    return df


df = feature_engineering(df)

train_df = df[df.dataset == 1]

# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
random.seed(42)


def custom_train_test_split(df, ratio=0.8, split=True):

    users = list(zip(df["userID"].value_counts().index, df["userID"].value_counts()))
    random.shuffle(users)

    max_train_data_len = ratio * len(df)
    sum_of_train_data = 0
    user_ids = []

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)

    train = df[df["userID"].isin(user_ids)]
    test = df[df["userID"].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test["userID"] != test["userID"].shift(-1)]
    return train, test

# 유저별 분리
train, test = custom_train_test_split(train_df)

# TODO :사용할 Feature 설정
FEATS = [
    "assessmentItemID",
    "testId",
    "KnowledgeTag",
    "user_acc",
    "user_total_answer",
    "test_mean",
    "test_sum",
    "tag_mean",
    "tag_sum",
    # -- 여기서부터 Custom Feature Engineering
    "bigClass",
    "bigClassAcc",
    "bigClassAccCate",
    "cumAccuracy",
    "cumCorrect",
    "elapsedTime",
    "elapsedTimeClass",
    # "KnowledgeTagAcc",
    # "KTAccuracyCate",
    "recAccuracy",
    "seenCount",
    "tagCluster",
    "tagCount",
    "testLV",
    "userLVbyTest",
    "year",
    "month",
    "day",
]

# X, y 값 분리
y_train = train["answerCode"]
train = train.drop(["answerCode"], axis=1)

y_test = test["answerCode"]
test = test.drop(["answerCode"], axis=1)

lgb_train = lgb.Dataset(train[FEATS], y_train)
lgb_test = lgb.Dataset(test[FEATS], y_test)

# hyper parameters
# TODO : tunning
params = {
    "learning_rate": 0.005,
    "max_depth": 13,
    "boosting": "dart",  # rf, gbdt, dart, goss
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 60,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,  # random forest면 0초과 1미만
    "bagging_freq": 5,  # random forest면 0 초과
    "seed": 42,
}


model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_test],
    verbose_eval=100,
    num_boost_round=100000,
    early_stopping_rounds=100,
)

preds = model.predict(test[FEATS])
acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
auc = roc_auc_score(y_test, preds)

print(f"VALID AUC : {auc} ACC : {acc}\n")

# INSTALL MATPLOTLIB IN ADVANCE
_ = lgb.plot_importance(model)

test_df = df[df.dataset == 2]

# LEAVE LAST INTERACTION ONLY
test_df = test_df[test_df["userID"] != test_df["userID"].shift(-1)]

# DROP ANSWERCODE
test_df = test_df.drop(["answerCode"], axis=1)
# MAKE PREDICTION
total_preds = model.predict(test_df[FEATS])
# SAVE OUTPUT
output_dir = "output/"
write_path = os.path.join(output_dir, "submission_dart.csv")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, "w", encoding="utf8") as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(total_preds):
        w.write("{},{}\n".format(id, p))