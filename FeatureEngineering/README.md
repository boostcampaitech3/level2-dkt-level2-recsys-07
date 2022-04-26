# Feature Engineering

userID : 유저ID

assessmentItemID : 문제번호

testId : 시험지번호

answerCode : 정답여부

Timestamp : 문제를 풀기 시작한 시각

KnowledgeTag : 문제 분류 태그

bigClass : 대분류 @jongmoon 

elapsedTime : 문제를 푸는데 걸린 시간 @jongmoon 

elapsedTimeClass : 문제를 푸는데 걸린 시간을 클래스로 분류 @jongmoon 

testLV : 시험지 난이도 @김소미 

tagLV : 태그 난이도 @김소미 

userLVbyTest : 시험지 난이도로 분류한 유저 실력 @김소미 

userLVbyTag : 태그 난이도로 분류한 유저 실력 @김소미 

bigClassAnswerRate : 대분류의 정답률 @David Seo 

cumAccuracy : 누적 정답률 @승주 백

recAccuracy : 최근 정답률 @승주 백

cumCorrect : 누적 정답 수 @승주 백

tagCount : 누적 태그의 수 @승주 백 

seenCount : 해당 문제를 이전에 몇 번 보았는가  @채오이 

KTAccuracy : knowledge tag별 개인당 정답률 @David Seo 

tagCluster : knowlege tag clustering @채오이 

userCluster : 사용자의 cluster @David Seo  @승주 백

# Feature Selector

    feature_selector : 실험 시 사용할 feature를 가지고있는 csv 파일을 만들 때 사용할 수 있습니다.