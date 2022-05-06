import pandas as pd
import numpy as np
import os
import random

import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import eli5
from eli5.sklearn import PermutationImportance

# warning 문구 무시
import warnings
warnings.filterwarnings(action="ignore")

from Wandb import CustomWandb
from args import parse_args
from IPython.display import display
from sklearn.model_selection import KFold


random.seed(42)

# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def option1_train_test_split(df, ratio=0.8, split=True):

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


def option2_train_test_split(df):
    # use train dataset only for train
    train = df[df.dataset == 1]

    # use test dataset only for valid
    test = df[(df.dataset == 2) & (df.answerCode != -1)]  # -1 인 answerCode 제외

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test["userID"] != test["userID"].shift(-1)]

    return train, test


def feature_engineering(df):
    # 유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # 카테고리형 feature
    categories = [
        "assessmentItemID",
        "testId",
    ]  # TODO : category feature를 변환시켜줘야함

    # label encode your categorical columns
    le = preprocessing.LabelEncoder()
    for category in categories:
        df[category] = le.fit_transform(df[category])
    return df


args = parse_args()
####################### Set Wandb Config #####################
wandb = CustomWandb(args)
wandb.config()
##############################################################

data_dir = "/opt/ml/input/data/"  # 경로는 상황에 맞춰서 수정해주세요!
csv_file_path = os.path.join(data_dir, "all_feature_data.csv")  # 데이터는 대회홈페이지에서 받아주세요 :)
df = pd.read_csv(csv_file_path)
df = feature_engineering(df)


# hyper parameters
# TODO : tunning
params = {
    "learning_rate": args.learning_rate,  # default = 0.1, [0.0005 ~ 0.5]
    # "max_depth": args.max_depth, # default = -1 (= no limit)
    "boosting": "gbdt",
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
# TODO :사용할 Feature 설정
FEATS = [
    "assessmentItemID",
    "testId",
    "KnowledgeTag",
    # "user_acc",
    # "user_total_answer",
    # "test_mean",
    # "test_sum",
    # "tag_mean",
    # "tag_sum",
    # -- 여기서부터 Custom Feature Engineering
    "bigClass",
    "bigClassAcc",
    "bigClassAccCate",
    # "recAccuracy",
    "cumAccuracy",
    "cumCorrect",
    "day",
    "month",
    "year",
    "elapsedTime",
    "elapsedTimeClass",
    # "KnowledgeTagAcc",
    # "KTAccuracyCate",
    "seenCount",
    "tagCluster",
    "tagCount",
    # "testLV",
    # "userLVbyTest",
    "userLVbyTestAVG",
    # "tagLV",
    # "userLVbyTag",
    "userLVbyTagAVG",
]

train_df = df[(df.dataset == 1)]
#train, valid = option1_train_test_split(train_df) # 유저별 분리

n_splits = 5
kfold_auc_list = list()
kf = KFold(n_splits=n_splits)

for k_th, (train_idx, valid_idx) in enumerate(kf.split(train_df)):

    train = train_df.iloc[train_idx]
    valid = train_df.iloc[valid_idx]
    
    # X, y 값 분리
    y_train = train["answerCode"]
    train = train.drop(["answerCode"], axis=1)

    y_valid = valid["answerCode"]
    valid = valid.drop(["answerCode"], axis=1)
    

    model = lgb.LGBMClassifier(
    **params,
    n_estimators=10000,
    silent=-1,
    )

    model.fit(
        train[FEATS],
        y_train,
        early_stopping_rounds=100,
        eval_set=[(train[FEATS], y_train), (valid[FEATS], y_valid)],
        eval_names=["train", "valid"],
        eval_metric="roc_auc",
        verbose=100,
    )

    preds = model.predict_proba(valid[FEATS])[:, 1]
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)


    perm = PermutationImportance(
        model, scoring="roc_auc", n_iter=1, random_state=42, cv=None, refit=False
    ).fit(valid[FEATS], y_valid)
    display(eli5.show_weights(perm, top=len(FEATS), feature_names=FEATS))

    
    perm_imp_df = pd.DataFrame()
    perm_imp_df["feature"] = FEATS
    perm_imp_df["importance"] = perm.feature_importances_
    perm_imp_df["std"] = perm.feature_importances_std_
    perm_imp_df.sort_values(by="importance", ascending=False, inplace=True)
    perm_imp_df.reset_index(drop=True, inplace=True)

    ########### Log Wandb K_th-fold Permutation Importance & Valid metrics ###########
    wandb.plot_importance(model, k_th+1)
    wandb.plot_perm_imp(perm_imp_df, k_th+1)
    wandb.table_perm_imp(perm_imp_df, k_th+1)
    
    metric = {"Valid/roc_auc": auc,
                "Valid/accuracy": acc}
    wandb.log(metric)
    ##################################################################################
    print(f"VALID AUC : {auc} ACC : {acc}\n")
    kfold_auc_list.append(auc)

kfold_auc = sum(kfold_auc_list) / n_splits
wandb.log({"kfold_auc": kfold_auc})
wandb.finish()