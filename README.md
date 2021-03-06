![image](https://user-images.githubusercontent.com/91870042/168077277-9bd49bcb-cb71-433b-9cf0-24f334b1a515.png)

# ๐ฏ Deep Knowledge Tracing

## ๐ ๋ํ ๊ฐ์

์ด๋ฑํ๊ต, ์คํ๊ต, ๊ณ ๋ฑํ๊ต, ๋ํ๊ต์ ๊ฐ์ ๊ต์ก๊ธฐ๊ด์์ ์ฐ๋ฆฌ๋ ์ํ์ ๋ ๋ด์์ต๋๋ค. ์ํ ์ฑ์ ์ด ๋์ ๊ณผ๋ชฉ์ ์ฐ๋ฆฌ๊ฐ ์ ์๋ ๊ฒ์ ๋ํ๋ด๊ณ  ์ํ ์ฑ์ ์ด ๋ฎ์ ๊ณผ๋ชฉ์ ๋ฐ๋๋ก ๊ณต๋ถ๊ฐ ๋์ฑ ํ์ํจ์ ๋ํ๋๋๋ค. ์ํ์ ์ฐ๋ฆฌ๊ฐ ์ผ๋ง๋งํผ ์๋์ง ํ๊ฐํ๋ ํ ๋ฐฉ๋ฒ์๋๋ค.

ํ์ง๋ง ์ํ์๋ ํ๊ณ๊ฐ ์์ต๋๋ค. ์ฐ๋ฆฌ๊ฐ ์ํ ์ํ์์ ์ ์๋ฅผ 80์  ๋ฐ์๋ค๋ฉด ์ฐ๋ฆฌ๋ 80์ ์ ๋ฐ์ ํ์์ผ ๋ฟ์๋๋ค. ์ฐ๋ฆฌ๊ฐ ๋์ ๋ค์ฌ ๊ณผ์ธ๋ฅผ ๋ฐ์ง ์๋ ์ด์ ์ฐ๋ฆฌ๋ ์ฐ๋ฆฌ ๊ฐ๊ฐ์ธ์ ๋ง์ถคํ๋ ํผ๋๋ฐฑ์ ๋ฐ๊ธฐ๊ฐ ์ด๋ ต๊ณ  ๋ฐ๋ผ์ ๋ฌด์์ ํด์ผ ์ฑ์ ์ ์ฌ๋ฆด ์ ์์์ง ํ๋จํ๊ธฐ ์ด๋ ต์ต๋๋ค. ์ด๋ด ๋ ์ฌ์ฉํ  ์ ์๋ ๊ฒ์ด DKT์๋๋ค!

DKT๋ Deep Knowledge Tracing์ ์ฝ์๋ก ์ฐ๋ฆฌ์ "์ง์ ์ํ"๋ฅผ ์ถ์ ํ๋ ๋ฅ๋ฌ๋ ๋ฐฉ๋ฒ๋ก ์๋๋ค.

## ๐ก ๋ชจ๋ธ

### 1. Transformer Based

- LSTM
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

## ๐ ๋๋ ํ ๋ฆฌ ๊ตฌ์กฐ

<details>
<summary>ํ๋ก์ ํธ๊ตฌ์กฐ ํผ์น๊ธฐ</summary>
<div markdown="1">

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

</div>
</details>

## ๐ ์์ธ ์ค๋ช

### 1. Feature Engineering

[๐ README: Feature Engineering](./FeatureEngineering/README.md)

### 2. Model

[๐ README: Model](./model/README.md)

### 3. Ensemble

[๐ README: Ensemble](./ensemble/README.md)

## ๐งช ์คํ ๊ด๋ฆฌ

- GitHub : https://github.com/boostcampaitech3/level2-dkt-level2-recsys-07
- Confluence : https://somi198.atlassian.net/wiki/spaces/DKT/pages
- Jira : https://somi198.atlassian.net/jira/software/projects/DKT/boards/1/roadmap

## ๐ฅ ์ต์ข ์์ ๋ฐ ๊ฒฐ๊ณผ
 
![image](https://user-images.githubusercontent.com/91870042/168078928-b55e74ef-cb6c-46eb-ab3c-2c79c8ae0bc8.png)
์ ์ฒด 16ํ ์ค 2์
