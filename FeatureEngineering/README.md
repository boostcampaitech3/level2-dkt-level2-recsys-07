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

`Grade` :  
`KTAccuracy` : 유저의 KnowledgeTag별 정답률  
`KTAccuracy_fixed`  
`RepeatedTime` :  
`accuracy` : 유저의 정답률  
`bigClass` : 문제의 대분류  
`bigClassAnswerRate` : 유저의 대분류별 정답률  
`bigClassCount` : 유저의 대분류 풀이 횟수  
`bigClassElapsedTimeAvg` : 유저의 대분류별 문제 풀이 시간 평균  
`elapsedTime` : 유저가 문제를 푸는데 걸린 시간 (Timestamp : 문제 풀이 시작시간 기준)  
`elapsedTimeClass` : 유저가 문제를 푸는데 걸린 시간을 Class로 분류  
`elapsedTime_ver2` : 유저가 문제를 푸는데 걸린 시간 (Timestamp : 문제 풀이 종료시간 기준)  
`elo` : elo function을 사용한 문제 난이도 계산  
`eloTag` : elo function을 사용한 KnowledgeTag 난이도 계산  
`eloTest` : elo function을 사용한 시험지 난이도 계산  
`feature_engineering`
`problemNumber` : 문제의 번호  
`recCount` :  
`relativeAnswerCode` :  
`seenCount` : 해당 문제를 이전에 몇 번 풀었는지 기록  
`tag_mean` :  
`test_mean` :  
`tagCluster` : 사용자의 cluster  
`userClustering` :  
`userLVbyTag` : 태그 난이도로 분류한 유저 실력  
`userLVbyTest` : 시험지 난이도로 분류한 유저 실력  
`wday` :  
`weekNum` :  
`hour` : Timestamp의 시간  
`yearMonthDay` : Timestamp의 날짜 추출

---

testLV : 시험지 난이도 @김소미

tagLV : 태그 난이도 @김소미

userLVbyTest : 시험지 난이도로 분류한 유저 실력 @김소미

userLVbyTag : 태그 난이도로 분류한 유저 실력 @김소미

bigClassAnswerRate : 대분류의 정답률 @David Seo

cumAccuracy : 누적 정답률 @승주 백

recAccuracy : 최근 정답률 @승주 백

cumCorrect : 누적 정답 수 @승주 백

tagCount : 누적 태그의 수 @승주 백

seenCount : 해당 문제를 이전에 몇 번 보았는가 @채오이

KTAccuracy : knowledge tag별 개인당 정답률 @David Seo

tagCluster : knowlege tag clustering @채오이

userCluster : 사용자의 cluster @David Seo @승주 백

# Feature Selector

    feature_selector : 실험 시 사용할 feature를 가지고있는 csv 파일을 만들 때 사용할 수 있습니다.
