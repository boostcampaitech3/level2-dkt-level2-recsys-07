# 문제풀이수가 너무 적은 유저를 삭제
import pandas as pd
import numpy as np
from tqdm import tqdm

train_df=pd.read_csv("/opt/ml/input/data/train_data.csv")
userID=pd.unique(train_df.userID).tolist()
user_ac=train_df.groupby("userID")["answerCode"].apply(list)

answer_rate_by_user=np.array(userID)
answer_rate_by_user=answer_rate_by_user[:, np.newaxis]
answer_rate_by_user=answer_rate_by_user.tolist()

for u in range(len(user_ac)):
    answer_rate_by_user[u].append(sum(user_ac[userID[u]])/len(user_ac[userID[u]]))
    answer_rate_by_user[u].append(len(user_ac[userID[u]]))

answer_rate_by_user_df=pd.DataFrame(data=answer_rate_by_user, index=userID, columns=['userID','answerRate','문제풀이수'])

small_problem_userID=answer_rate_by_user_df[answer_rate_by_user_df['문제풀이수']<=30].userID.tolist() # 제거해야할 user ID들 # 30개 이하인 user

# outlier_userID=[5887, 6283, 6382, 6764, 7029, 7166, 5498, 5820, 6760, 6988, 7171, 7186, 481] # user_answer_rate에 해당하는 outlier user ID
# cnt=0
# for out in outlier_userID:
#     if out in small_problem_userID:
#         cnt+=1
# print(f'user_answer_rate outlier와 겹치는 id 갯수 :{cnt}')

print('원래 길이: ', len(train_df))
for outlier in tqdm(small_problem_userID): # 3분정도 소요
    drop_idx=train_df[train_df['userID'] == outlier].index
    train_df.drop(inplace=True, axis=0, index=drop_idx)
print('제거 후 길이: ', len(train_df))

train_df.to_csv('/opt/ml/input/data/train_data_small_solved_problem.csv')
print('문제풀이수가 적은 유저가 제거된 train_data_small_solved_problem.csv 생성 완료')