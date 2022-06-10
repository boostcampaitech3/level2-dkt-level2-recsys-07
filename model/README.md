# Introduction

Deep Knowledge Tracing 대회에서는 총 4분류의 모델을 사용하였습니다. 각 모델들은 별도의 폴더로 분류되어 있으며, 폴더내에 비슷한 계열의 모델들 끼리 묶어놓았습니다.

1. Boosting Based
    - XGBoost
    - LGBM
    - CatBoost

2. Graph Based
    - Light GCN

3. Rule Based
    - Main Category

4. Transformer Based
    - LSTM
    - LSTM Attention
    - BERT
    - Last Query Transformer
    - Saint 

<br>

## Model Experiments

preprocessing 을 통해서 만들어진 약 40개의 Feature를 사용해서 모델을 학습을 진행하였습니다. 일부 모델(LGCN)은 전체 Feature 정보를 사용하지 않고, Label 값만 사용해서 학습을 진행합니다.

1. Sequence / Transformer
    ```
    python3 train.py
    python3 inference.py
    ```

    필요한 hyperparameter 또는 path arguments 들은 `args.py` 를 통해서 수정이 가능합니다. `--model` 인자에는 다음과 같은 값들을 선택하여 모델을 달리하여 실험할 수 있습니다.

    model arguments : **lstm, lstmattn, bert, LastQuery, Saint**

2. Boosting

    ```
    CatBoost.ipynb
    LGBM.ipynb
    XGBoost.ipynb
    ```

    각 노트북 파일을 통해서 원하는 Ensemble Model 을 실행시킬 수 있습니다. 각 모델에 입력으로 사용되는 데이터의 Feature가 다를 경우, 성능의 차이가 발생할 수 있습니다.

3. Graph

    ```
    pip install -r requirements.txt
    python3 train.py
    python3 inference.py
    ```

    python package의 종속성 영향을 많이 받게 되므로 `requirements.txt` 를 통한 패키지 관리가 필수적으로 요구됩니다.

4. Rule Based

    ```
    python3 MainCategoryRuleBased.py
    ```

    위의 파일을 실행시키면, 동일 폴더내에 정답파일인 `csv` 파일이 생성됩니다. 문제의 태그에 의한 정답률을 Base로 하였고, 규칙기반으로 정오답 여부를 예측하도록 하였습니다.

<br>

## Model Results

|Model|Augmentation / Skills | AUROC | Accuracy |
|:---:|:---:|:---:|:---:|
|LSTM|Sweep|0.7581|-|
|LSTM Attention|Sliding Window, Continous Feature, K-Fold|0.8499|0.7849|
|Last Query Transformer|Sliding Window, Continous Feature, K-Fold|0.8467|0.7399|
|BERT|Sliding Window, Continous Feature, K-Fold|0.8212|0.7366|
|Saint|Sliding Window, Continous Feature|-|-|
|LGBM|Feature Selection / Engineering | 0.8400 | 0.7419|
|XGBoost|Feature Selection / Engineering|0.8571|0.7715|
|CatBoost|Feature Selection / Engineering|0.8504|0.7608|
|Light GCN|-|0.8425|0.7608|