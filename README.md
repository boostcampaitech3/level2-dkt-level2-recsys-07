![image](https://user-images.githubusercontent.com/91870042/168077277-9bd49bcb-cb71-433b-9cf0-24f334b1a515.png)

# 💯 Deep Knowledge Tracing

## 🏆 대회 개요

초등학교, 중학교, 고등학교, 대학교와 같은 교육기관에서 우리는 시험을 늘 봐왔습니다. 시험 성적이 높은 과목은 우리가 잘 아는 것을 나타내고 시험 성적이 낮은 과목은 반대로 공부가 더욱 필요함을 나타냅니다. 시험은 우리가 얼마만큼 아는지 평가하는 한 방법입니다.

하지만 시험에는 한계가 있습니다. 우리가 수학 시험에서 점수를 80점 받았다면 우리는 80점을 받은 학생일 뿐입니다. 우리가 돈을 들여 과외를 받지 않는 이상 우리는 우리 개개인에 맞춤화된 피드백을 받기가 어렵고 따라서 무엇을 해야 성적을 올릴 수 있을지 판단하기 어렵습니다. 이럴 때 사용할 수 있는 것이 DKT입니다!

DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

## 💡 모델

### 1. Transformer Based

- LSTM Attention
- BERT
- Saint
- Last Query Transformer

### 2. Boosting Based

- LGBM: Light Gradient Boosting Machine
- CatBoost
- XGBoost

### 3. Rule Based

- KT by problem Class

### 4. Graph Based

- Light GCN

## 📂 디렉토리 구조

```
.
|-- EDA
|   |-- EDA_by_tag.ipynb
|   |-- Student_score_analysis.ipynb
|   |-- TestId_EDA.ipynb
|   |-- eda_Jongmoon.ipynb
|   `-- tNSE.ipynb
|-- FeatureEngineering
|   |-- Grade.ipynb
|   |-- KTAccuracy.ipynb
|   |-- KTAccuracy_fixed.ipynb
|   |-- README.md
|   |-- RepeatedTime.ipynb
|   |-- accuracy.ipynb
|   |-- bigClass.ipynb
|   |-- bigClassAnswerRate.ipynb
|   |-- bigClassCount.ipynb
|   |-- bigClassElapsedTimeAvg.ipynb
|   |-- elapsedTime.ipynb
|   |-- elapsedTimeClass.ipynb
|   |-- elapsedTime_ver2.ipynb
|   |-- elo.ipynb
|   |-- eloTag.ipynb
|   |-- eloTest.ipynb
|   |-- feature_engineering.ipynb
|   |-- feature_selector.ipynb
|   |-- prev_1st_FE.ipynb
|   |-- problemNumber.ipynb
|   |-- recCount.ipynb
|   |-- relativeAnswerCode.ipynb
|   |-- seenCount.ipynb
|   |-- split_train_test.ipynb
|   |-- tag&test_mean.ipynb
|   |-- tagCluster.ipynb
|   |-- userClustering.ipynb
|   |-- userLVbyTag.py
|   |-- userLVbyTest.py
|   |-- wday,weekNum,hour.ipynb
|   `-- yearMonthDay.ipynb
|-- README.md
|-- ensemble
|   |-- average_ensemble.ipynb
|   |-- ensemble.png
|   |-- ensemble_result
|   |-- hard_soft_ensemble.ipynb
|   `-- output_files
|-- model
|   |-- Boosting\ based
|   |   |-- CatBoost
|   |   |   `-- CatBoost.ipynb
|   |   |-- LGBM
|   |   |   |-- LGBM.ipynb
|   |   |   |-- LGBM_jupyterLab.py
|   |   |   |-- LGBM_ver2.ipynb
|   |   |   |-- NewLGBM.ipynb
|   |   |   |-- output
|   |   |   `-- sweep
|   |   |       |-- LGBM.py
|   |   |       |-- LGBM_ver2.py
|   |   |       |-- Wandb.py
|   |   |       |-- args.py
|   |   |       `-- sweep.yaml
|   |   `-- XGBoost
|   |       |-- XGBoost.ipynb
|   |       `-- output
|   |-- Graph\ based
|   |   `-- LightGCN
|   |       |-- README.md
|   |       |-- config.py
|   |       |-- inference.py
|   |       |-- lightgcn
|   |       |   |-- datasets.py
|   |       |   |-- models.py
|   |       |   `-- utils.py
|   |       |-- output
|   |       |-- sweep.yaml
|   |       |-- train.py
|   |       `-- weight
|   |-- RuleBased
|   |   `-- MainCategoryRuleBased.py
|   `-- Transformer\ based
|       |-- DKT_Baseline
|       |   |-- README.md
|       |   |-- args.py
|       |   |-- dkt
|       |   |   |-- criterion.py
|       |   |   |-- data
|       |   |   |   `-- data_timeElapsed.csv
|       |   |   |-- dataloader.py
|       |   |   |-- metric.py
|       |   |   |-- model.py
|       |   |   |-- optimizer.py
|       |   |   |-- scheduler.py
|       |   |   |-- trainer.py
|       |   |   `-- utils.py
|       |   |-- inference.py
|       |   |-- requirements.txt
|       |   |-- sweep.yaml
|       |   `-- train.py
|       `-- dktWithFeatures
|           |-- args.py
|           |-- dkt
|           |   |-- criterion.py
|           |   |-- dataloader.py
|           |   |-- metric.py
|           |   |-- model.py
|           |   |-- optimizer.py
|           |   |-- pseudoLabelTrainer.py
|           |   |-- scheduler.py
|           |   |-- trainer.py
|           |   `-- utils.py
|           |-- inference.py
|           |-- models
|           |-- output
|           `-- train.py
`-- preprocessing
    |-- split_FE_dataset.ipynb
    |-- split_train_test_set.ipynb
    |-- tabular.ipynb
    |-- train_small_solved_problem.py
    `-- train_user_answer_rate.py
```

## 🧪 실험 관리

- GitHub : https://github.com/boostcampaitech3/level2-dkt-level2-recsys-07
- Confluence : https://somi198.atlassian.net/wiki/spaces/DKT/pages
- Jira : https://somi198.atlassian.net/jira/software/projects/DKT/boards/1/roadmap

## 🥈 최종 순위 및 결과

![image](https://user-images.githubusercontent.com/91870042/168078928-b55e74ef-cb6c-46eb-ab3c-2c79c8ae0bc8.png)
전체 16팀 중 2위
