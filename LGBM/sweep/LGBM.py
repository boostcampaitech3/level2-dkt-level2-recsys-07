import pandas as pd
import numpy as np
import os
import random

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# warning 문구 무시
import warnings

warnings.filterwarnings(action="ignore")

import wandb
from args import parse_args
from sklearn.model_selection import KFold

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


wandb.login()
args = parse_args()

data_dir = "/opt/ml/input/data/"  # 경로는 상황에 맞춰서 수정해주세요!
csv_file_path = os.path.join(
    data_dir, "FE_total_data_with_bigClassAccCate.csv"
)  # 데이터는 대회홈페이지에서 받아주세요 :)
df = pd.read_csv(csv_file_path)

categories = ["assessmentItemID", "testId"]

for category in categories:
    df[category] = df[category].astype("category")

# hyper parameters
# TODO : tunning
params = {
    "learning_rate": args.learning_rate,  # default = 0.1, [0.0005 ~ 0.5]
    # "max_depth": args.max_depth, # default = -1 (= no limit)
    "boosting": "rf",
    "objective": args.objective,
    "metric": args.metric,
    "num_leaves": args.num_leaves,  # default = 31, [10, 20, 31, 40, 50]
    "feature_fraction": args.feature_fraction,  # default = 1.0, [0.4, 0.6, 0.8, 1.0]
    "bagging_fraction": args.bagging_fraction,  # default = 1.0, [0.4, 0.6, 0.8, 1.0]
    "bagging_freq": args.bagging_freq,  # default = 0, [0, 1, 2, 3, 4]
    "seed": 42,
    "verbose": -1,
}

# TODO :사용할 Feature 설정
FEATS = [
    "assessmentItemID",
    "testId",
    "KnowledgeTag"
    #'bigClassAccCate'
    #'user_total_answer',
    #'test_mean',
    #'test_sum',
    #'tag_mean',
    #'tag_sum',
    #'cumCorrect',
    #'cumAccuracy',
    #'tagCount',
    #'recAccuracy'
]

train_df = df[df.dataset == 1]

# 유저별 분리
# train, test = custom_train_test_split(train_df)

n_splits = 5
kfold_auc_list = list()
kf = KFold(n_splits=n_splits)
for k_th, (train_idx, valid_idx) in enumerate(kf.split(train_df)):

    train = train_df.iloc[train_idx]
    test = train_df.iloc[valid_idx]
    # X, y 값 분리
    y_train = train["answerCode"]
    train = train.drop(["answerCode"], axis=1)

    y_test = test["answerCode"]
    test = test.drop(["answerCode"], axis=1)

    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_test = lgb.Dataset(test[FEATS], y_test)

    wandb.init(project="my_test_project")

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        verbose_eval=-1,
        num_boost_round=10000,
        early_stopping_rounds=50,
    )

    preds = model.predict(test[FEATS])
    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)

    wandb.log({"valid_auc": auc, "valid_acc": acc})
    print(f"VALID AUC : {auc} ACC : {acc}\n")
    kfold_auc_list.append(auc)

kfold_auc = sum(kfold_auc_list) / n_splits
wandb.log({"kfold_auc": kfold_auc})
