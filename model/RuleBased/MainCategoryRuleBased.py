import numpy as np
import pandas as pd
import tqdm

variable = "KnowledgeTag" # "main", "KnowledgeTag" 가능. main = 대분류, KnowledgeTag = 중분류

print("Loading Data..")
raw_data = pd.read_csv("/opt/ml/input/data/test_data.csv")
# 원래 데이터에 대분류 열을 추가 -> 대분류 = 시험지 번호 앞 3자리 중 중간 자리 수
raw_data.insert(loc=6, column="main", value=0)
for i in tqdm.tqdm(range(len(raw_data["testId"]))) :
    raw_data["main"].iloc[i] = raw_data["testId"].iloc[i][2]

# 우리가 예측해야 하는 유저의 id와 문제의 대분류 prediction 리스트에 저장
prediction = list()
for i in range(len(raw_data["userID"])) :
    if raw_data["answerCode"].iloc[i] == -1 :
        prediction.append((raw_data["userID"].iloc[i], raw_data[variable].iloc[i])) # variable안에 우리가 지정한 변수가 들어가고, 이를 추출한다
print("Data Loaded and Preprocessed!")

# 예측해야 되는 문제의 정답률 계산
print("Calculating Results...")
result = list()
cnt = 0
for user in tqdm.tqdm(prediction) :
    v = user[1] # user[2] = 대분류
    id = user[0] 
    total = 0
    correct = 0
    data = raw_data[raw_data["userID"]==id] # 유저별로 과거 풀이 데이터 추출
    for i in range(len(data)) :
        if data[variable].iloc[i]==v: # 만약 문제가 현재 푸는 문제와 variable이 동일하다면
            if data["answerCode"].iloc[i] == 1 : # 만약 문제를 맞췄다면
                correct +=1 # 정답 개수 + 1
                total +=1 # 전체 개수 + 1
                continue
            elif data["answerCode"].iloc[i] == 0 : #만약 문제를 틀렸다면
                total += 1 # 전체 개수만 + 1
    if total == 0: # KnowledgeTag를 사용하는 경우 같은 경우의 수가 없어서 분모가 0이 될 수 있다. DivisionError를 방지하기 위해 추가.
        result.append((cnt, 0)) # 만약 같은 분류의 문제를 풀어본 적이 없다면, 틀렸다고 가정
    else : result.append((cnt, correct/total)) 
    cnt += 1

    
# 파일 저장
pd.DataFrame(data=result, columns=["id","prediction"]).to_csv("rulebased.csv", index=0)
print("Inference Complete!")