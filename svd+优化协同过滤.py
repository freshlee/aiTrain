import numpy as np
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
moviesDF = pd.read_csv(moviesPath, index_col=None, nrows=600)
# 数据量有点大。机子不好的同学使用nrows调整数据读取条目
ratingsDF = pd.read_csv(ratingsPath, index_col=None, nrows=600)
# print(ratingsDF[['userId', 'rating', 'movieId']])
# print(ratingsDF.describe())
u_m4Rating = np.array(pd.pivot_table(ratingsDF,index=["userId"], columns=['movieId'], values='rating')) # .fillna(0):nan 转为0

user_total = len(u_m4Rating)
movie_total = len(u_m4Rating[0])

def calCosineSimilarity(list1, list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1, val2) in zip(list1, list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))

u_u4cos = np.zeros(shape=(user_total, user_total))
def filterNeverLook(list1, list2):
    list1_res = list1 = np.array(list1)
    list2_res = list2 = np.array(list2)
    for i in range(len(list1))[::-1]:
        if math.isnan(list1[i]) == True or math.isnan(list2[i]) == True:
            list1_res = np.delete(list1_res, i, axis = 0)
            list2_res = np.delete(list2_res, i, axis = 0)
                
    return list1_res, list2_res
                
                
for i in range(user_total):
    for j in range(user_total):
        if math.isnan(u_u4cos[j][i]):
            u_u4cos[i][j] = u_u4cos[j][i]
        else:
            list1, list2 = filterNeverLook(u_m4Rating[i], u_m4Rating[j])
            # print(list1)
            if len(list1) > 0:
                u_u4cos[i][j] = calCosineSimilarity(list1, list2)

# u_m4Rating_cal = np.zeros(shape=u_m4Rating.shape)
# for i in u_m4Rating:
#     print(len(np.argwhere(np.isnan(i))))
a = np.array(list(len(np.argwhere(np.isnan(i))) for i in u_m4Rating))
u_m4Rating2zero = np.nan_to_num(u_m4Rating)
# print(a, u_m4Rating2zero)
u_m4Rating_cal = np.divide(np.matmul(np.transpose(u_m4Rating2zero), u_u4cos), a)
recommand_list = u_m4Rating_cal[:, 1]
# recommand_list = np.sort(recommand_list, axis=0)
recommand_list = sorted(range(len(recommand_list)), key=lambda i:recommand_list[i])
print(recommand_list[-10:])
moviesDF = np.array(moviesDF)
for i in recommand_list[-10:]:
    print(moviesDF[i])

# a = np.array([1 , 4, 6, None])

# a = np.delete(a,3, axis=0)
# print(a)

# print(u_u4cos)

#判断电影喜爱程度



# A =  np.eye(3)
# u,s,v = np.linalg.svd(A)
# print(u,s,v)