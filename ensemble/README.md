# Ensemble Methods

## 1. Ensemble by average

여러개의 제출 파일에 대해서 확률의 평균을 계산하는 방법

![image](https://user-images.githubusercontent.com/91870042/173056902-abfa6d6c-e6ca-4068-88a9-69db09bd5146.png)

## 2. Hard soft ensemble

문제를 맞출 확률 절반 50%를 기준으로 잘라서 더 많은 쪽의 확률에 대해 평균을 계산하는 방법

![ensemble](https://user-images.githubusercontent.com/91870042/173056979-89242942-fa6b-4087-a816-4f708c5803bd.png)


## Directory architecture

```
.
|-- README.md
|-- average_ensemble.ipynb
|-- ensemble.png
|-- ensemble_result
|-- hard_soft_ensemble.ipynb
`-- output_files
```

## How to use

1. 앙상블 하고자하는 submission.csv 파일들을 `output_files` 폴더 안에 모두 옮겨 넣습니다.
2. 원하는 앙상블 방법을 선택하고 해당 방법에 맞는 jupyter notebook 을 실행합니다.
3. 최종 앙상블된 결과는 `ensemble_result` 디렉토리 안에 저장됩니다.
