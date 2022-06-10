# Feature Engineering

```
.
|-- Grade.ipynb
|-- KTAccuracy.ipynb
|-- KTAccuracy_fixed.ipynb
|-- README.md
|-- RepeatedTime.ipynb
|-- accuracy.ipynb
|-- bigClass.ipynb
|-- bigClassAnswerRate.ipynb
|-- bigClassCount.ipynb
|-- bigClassElapsedTimeAvg.ipynb
|-- elapsedTime.ipynb
|-- elapsedTimeClass.ipynb
|-- elapsedTime_ver2.ipynb
|-- elo.ipynb
|-- eloTag.ipynb
|-- eloTest.ipynb
|-- feature_engineering.ipynb
|-- feature_selector.ipynb
|-- prev_1st_FE.ipynb
|-- problemNumber.ipynb
|-- recCount.ipynb
|-- relativeAnswerCode.ipynb
|-- seenCount.ipynb
|-- split_train_test.ipynb
|-- tag&test_mean.ipynb
|-- tagCluster.ipynb
|-- userClustering.ipynb
|-- userLVbyTag.py
|-- userLVbyTest.py
|-- wday,weekNum,hour.ipynb
`-- yearMonthDay.ipynb
```

## 🎲 Basic feature

`userID` : 유저ID  
`assessmentItemID` : 문제번호  
`testId` : 시험지번호  
`answerCode` : 정답여부  
`Timestamp` : 문제를 풀기 시작한 시각  
`KnowledgeTag` : 문제 분류 태그

## 🔧 feature engineering

### 통계
`tagMean` : 문제의 태그를 기준으로 한 정답률  
`tagSum` : 문제의 태그를 기준으로 누적 정답횟수  
`tagStd` : 문제의 태그를 기준으로한 정답여부 표준편차  
`testMean` : 문제를 기준으로 한 정답률  
`testSum` : 문제를 기준으로 한 누적 정답횟수  
`testStd` : 문제를 기준으로 한 정답여부 표준편차  

### 대분류
`bigClass` : 문제의 대분류  
`bigClassAcc` : 유저의 대분류별 정답률  
`bigClassAccCate` : 유저의 대분류별 정답률 categorical 화  
`bigClassCount` : 유저의 대분류 풀이 횟수   
`bigClassElapsedTimeAvg` : 유저의 대분류별 문제 풀이 시간 평균  

### answerCode
`recAccuracy` : 최근 정답률  
`recCount` :  최근 맞춘 정답 갯수  
`cumAccuracy` : 누적 정답률  
`cumCorrect` : 누적 정답 수  
`accuracy` : 유저의 정답률  
`totalAnswer` : 해당 문제를 맞춘 총 횟수를 계산
`seenCount` : 해당 문제를 이전에 몇 번 풀었는지 기록   
`relativeAnswerCode` : 상대적 정답코드(잘못된 feature)  

### timestamp
`day` : Timestamp의 날짜 추출  
`month` : Timestamp의 날짜 추출  
`year` :Timestamp의 날짜 추출  
`wday` :  요일  
`hour` : Timestamp의 시간    
`weekNum` : 주차  
`elapsedTime` : 유저가 문제를 푸는데 걸린 시간 (Timestamp : 문제 풀이 시작시간 기준)    
`elapsedTimeClass` : 유저가 문제를 푸는데 걸린 시간을 Class로 분류  
`elapsedTime_ver2` : 유저가 문제를 푸는데 걸린 시간 (Timestamp : 문제 풀이 종료시간 기준)  

### Knowledge Tag
`KTAccuracy` : 유저의 KnowledgeTag별 정답률 (knowledge tag별 개인당 정답률)  
`KTAccuracy_fixed` : 유저의 KnowledgeTag별 정답률
`KTAccuracyCate` : 유저의 KnowledgeTag별 정답률 카테고리화
`tagCluster` : knowlege tag clustering  
`tagCount` : 누적 태그의 수  
`userLVbyTag` : 태그 난이도로 분류한 유저 실력   
`userLVbyTagAVG` : 태그 난이도로 분류한 유저 실력의 평균  
`tagLV` : 태그 난이도  
`tagClass` : 태그의 난이도를 사용하여 Class로 분류

 ### testID
`testLV` : 시험지 난이도  
`userLVbyTest` : 시험지 난이도로 분류한 유저 실력  
`userLVbyTestAVG` : 시험지 난이도로 분류한 유저 실력의 평균  

### difficulty
`elo` : elo function을 사용한 문제 난이도 계산  
`eloTag` : elo function을 사용한 KnowledgeTag 난이도 계산  
`eloTest` : elo function을 사용한 시험지 난이도 계산  

`problemNumber` : 문제의 번호  
`userCluster` : 사용자의 cluster  

# Feature Selector
    feature_selector : 실험 시 사용할 feature를 가지고있는 csv 파일을 만들 때 사용할 수 있습니다.
