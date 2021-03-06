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

## ๐ฒ Basic feature

`userID` : ์ ์ ID  
`assessmentItemID` : ๋ฌธ์ ๋ฒํธ  
`testId` : ์ํ์ง๋ฒํธ  
`answerCode` : ์ ๋ต์ฌ๋ถ  
`Timestamp` : ๋ฌธ์ ๋ฅผ ํ๊ธฐ ์์ํ ์๊ฐ  
`KnowledgeTag` : ๋ฌธ์  ๋ถ๋ฅ ํ๊ทธ

## ๐ง feature engineering

### ํต๊ณ
`tagMean` : ๋ฌธ์ ์ ํ๊ทธ๋ฅผ ๊ธฐ์ค์ผ๋ก ํ ์ ๋ต๋ฅ   
`tagSum` : ๋ฌธ์ ์ ํ๊ทธ๋ฅผ ๊ธฐ์ค์ผ๋ก ๋์  ์ ๋ตํ์  
`tagStd` : ๋ฌธ์ ์ ํ๊ทธ๋ฅผ ๊ธฐ์ค์ผ๋กํ ์ ๋ต์ฌ๋ถ ํ์คํธ์ฐจ  
`testMean` : ๋ฌธ์ ๋ฅผ ๊ธฐ์ค์ผ๋ก ํ ์ ๋ต๋ฅ   
`testSum` : ๋ฌธ์ ๋ฅผ ๊ธฐ์ค์ผ๋ก ํ ๋์  ์ ๋ตํ์  
`testStd` : ๋ฌธ์ ๋ฅผ ๊ธฐ์ค์ผ๋ก ํ ์ ๋ต์ฌ๋ถ ํ์คํธ์ฐจ  

### ๋๋ถ๋ฅ
`bigClass` : ๋ฌธ์ ์ ๋๋ถ๋ฅ  
`bigClassAcc` : ์ ์ ์ ๋๋ถ๋ฅ๋ณ ์ ๋ต๋ฅ   
`bigClassAccCate` : ์ ์ ์ ๋๋ถ๋ฅ๋ณ ์ ๋ต๋ฅ  categorical ํ  
`bigClassCount` : ์ ์ ์ ๋๋ถ๋ฅ ํ์ด ํ์   
`bigClassElapsedTimeAvg` : ์ ์ ์ ๋๋ถ๋ฅ๋ณ ๋ฌธ์  ํ์ด ์๊ฐ ํ๊ท   

### answerCode
`recAccuracy` : ์ต๊ทผ ์ ๋ต๋ฅ   
`recCount` :  ์ต๊ทผ ๋ง์ถ ์ ๋ต ๊ฐฏ์  
`cumAccuracy` : ๋์  ์ ๋ต๋ฅ   
`cumCorrect` : ๋์  ์ ๋ต ์  
`accuracy` : ์ ์ ์ ์ ๋ต๋ฅ   
`totalAnswer` : ํด๋น ๋ฌธ์ ๋ฅผ ๋ง์ถ ์ด ํ์๋ฅผ ๊ณ์ฐ  
`seenCount` : ํด๋น ๋ฌธ์ ๋ฅผ ์ด์ ์ ๋ช ๋ฒ ํ์๋์ง ๊ธฐ๋ก   
`relativeAnswerCode` : ์๋์  ์ ๋ต์ฝ๋(์๋ชป๋ feature)  

### timestamp
`day` : Timestamp์ ๋ ์ง ์ถ์ถ  
`month` : Timestamp์ ๋ ์ง ์ถ์ถ  
`year` :Timestamp์ ๋ ์ง ์ถ์ถ  
`wday` :  ์์ผ  
`hour` : Timestamp์ ์๊ฐ    
`weekNum` : ์ฃผ์ฐจ  
`elapsedTime` : ์ ์ ๊ฐ ๋ฌธ์ ๋ฅผ ํธ๋๋ฐ ๊ฑธ๋ฆฐ ์๊ฐ (Timestamp : ๋ฌธ์  ํ์ด ์์์๊ฐ ๊ธฐ์ค)    
`elapsedTimeClass` : ์ ์ ๊ฐ ๋ฌธ์ ๋ฅผ ํธ๋๋ฐ ๊ฑธ๋ฆฐ ์๊ฐ์ Class๋ก ๋ถ๋ฅ  
`elapsedTime_ver2` : ์ ์ ๊ฐ ๋ฌธ์ ๋ฅผ ํธ๋๋ฐ ๊ฑธ๋ฆฐ ์๊ฐ (Timestamp : ๋ฌธ์  ํ์ด ์ข๋ฃ์๊ฐ ๊ธฐ์ค)  

### Knowledge Tag
`KTAccuracy` : ์ ์ ์ KnowledgeTag๋ณ ์ ๋ต๋ฅ  (knowledge tag๋ณ ๊ฐ์ธ๋น ์ ๋ต๋ฅ )  
`KTAccuracy_fixed` : ์ ์ ์ KnowledgeTag๋ณ ์ ๋ต๋ฅ   
`KTAccuracyCate` : ์ ์ ์ KnowledgeTag๋ณ ์ ๋ต๋ฅ  ์นดํ๊ณ ๋ฆฌํ  
`tagCluster` : knowlege tag clustering  
`tagCount` : ๋์  ํ๊ทธ์ ์  
`userLVbyTag` : ํ๊ทธ ๋์ด๋๋ก ๋ถ๋ฅํ ์ ์  ์ค๋ ฅ   
`userLVbyTagAVG` : ํ๊ทธ ๋์ด๋๋ก ๋ถ๋ฅํ ์ ์  ์ค๋ ฅ์ ํ๊ท   
`tagLV` : ํ๊ทธ ๋์ด๋  
`tagClass` : ํ๊ทธ์ ๋์ด๋๋ฅผ ์ฌ์ฉํ์ฌ Class๋ก ๋ถ๋ฅ  

 ### testID
`testLV` : ์ํ์ง ๋์ด๋  
`userLVbyTest` : ์ํ์ง ๋์ด๋๋ก ๋ถ๋ฅํ ์ ์  ์ค๋ ฅ  
`userLVbyTestAVG` : ์ํ์ง ๋์ด๋๋ก ๋ถ๋ฅํ ์ ์  ์ค๋ ฅ์ ํ๊ท   

### difficulty
`elo` : elo function์ ์ฌ์ฉํ ๋ฌธ์  ๋์ด๋ ๊ณ์ฐ  
`eloTag` : elo function์ ์ฌ์ฉํ KnowledgeTag ๋์ด๋ ๊ณ์ฐ  
`eloTest` : elo function์ ์ฌ์ฉํ ์ํ์ง ๋์ด๋ ๊ณ์ฐ  
`problemNumber` : ๋ฌธ์ ์ ๋ฒํธ  
`userCluster` : ์ฌ์ฉ์์ cluster  

# Feature Selector
    feature_selector : ์คํ ์ ์ฌ์ฉํ  feature๋ฅผ ๊ฐ์ง๊ณ ์๋ csv ํ์ผ์ ๋ง๋ค ๋ ์ฌ์ฉํ  ์ ์์ต๋๋ค.
