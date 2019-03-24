import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

moviesPath = ".\\data\\movies.csv"
ratingsPath = ".\\data\\ratings.csv"
moviesDF = pd.read_csv(moviesPath, index_col=None)
# 数据量有点大。机子不好的同学使用nrows调整数据读取条目
ratingsDF = pd.read_csv(ratingsPath, index_col=None, nrows=1000)

trainRatingsDF, testRatingsDF = train_test_split(ratingsDF, test_size=0.2)
# print(trainRatingsDF)
trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                 index=['userId'], values='rating', fill_value=0)

# enumerate返回穷举序列号与值
# 8981部电影
moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
# 610个用户
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
# 矩阵变成list 每一行变成list的一个值 长度为610 每个值大小为8981
ratingValues = trainRatingsPivotDF.values.tolist()
# print(ratingValues)

def calCosineSimilarity(list1, list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1, val2) in zip(list1, list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))
# 根据用户对电影的评分，来判断每个用户间相似度 len(ratingValues) => 矩阵有多少行，有多少用户
userSimMatrix = np.zeros((len(ratingValues), len(ratingValues)), dtype=np.float32)

# print(ratingValues)

for i in range(len(ratingValues) - 1):
    for j in range(i + 1, len(ratingValues)):
        userSimMatrix[i, j] = calCosineSimilarity(ratingValues[i], ratingValues[j])
        userSimMatrix[j, i] = userSimMatrix[i, j]
userMostSimDict = dict()
for i in range(len(ratingValues)):
    userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])), key=lambda x: x[1], reverse=True)[:10]
# 用这K个用户的喜好中目标用户没有看过的电影进行推荐
userRecommendValues = np.zeros((len(ratingValues), len(ratingValues[0])), dtype=np.float32)  # 610*8981

# print(len(ratingValues), len(ratingValues[0]))

for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):
        if ratingValues[i][j] == 0:
            val = 0
            for (user, sim) in userMostSimDict[i]:
                val += (ratingValues[user][j] * sim)
            userRecommendValues[i, j] = val
userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])), key=lambda x: x[1], reverse=True)[:10]
# 将一开始的索引转换为原来用户id与电影id
userRecommendList = []
userRecommendDictAll = dict()
for key, value in userRecommendDict.items():
    user = usersMap[key]
    for (movieId, val) in value:
        userRecommendList.append([user, moviesMap[movieId]])
        userRecommendDictAll['uid_' + str(user) + ';mid_' + str(moviesMap[movieId])] = val
        # 建立测试索引
print(userRecommendDictAll)
# 将推荐结果的电影id转换成对应的电影名
recommendDF = pd.DataFrame(userRecommendList, columns=['userId', 'movieId'])
recommendDF = pd.merge(recommendDF, moviesDF[['movieId', 'title']], on='movieId', how='inner')
# print(recommendDF.tail(10))

# 计算准确度
# print(userRecommendList, testRatingsDF)
testTable = pd.pivot_table(testRatingsDF[['userId', 'movieId', 'rating']], index=['userId'], columns=['movieId'], values=['rating'])

# 电影
moviesMap = dict(enumerate(list(testTable.columns)))
# 用户
usersMap = dict(enumerate(list(testTable.index)))

valueMatrx = testTable.values.tolist()
sum1=[]
sum2=[]
for userIndex in range(len(valueMatrx)):
        # print(valueMatrx[userIndex])
        for movieIndex in range(len(valueMatrx[userIndex])):
                user = str(usersMap[userIndex])
                movie = str(moviesMap[movieIndex][1])
                key = 'uid_' + user + ';mid_' + movie
                if (key in userRecommendDictAll.keys() and not pd.isnull(valueMatrx[userIndex][movieIndex])):
                        sum1.append(userRecommendDictAll[key])
                        sum2.append(valueMatrx[userIndex][movieIndex])
                        # print(key, userRecommendDictAll[key], valueMatrx[userIndex][movieIndex])
var = pearsonr(sum1, sum2)
print(var)