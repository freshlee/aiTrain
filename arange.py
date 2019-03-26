import calendar
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
def formate(lst):
    mission = pd.DataFrame(lst,
        columns=['项目', '天数', 'name'],
        index=lst[:, 2])
    # print(mission)
    # lstg = mission.groupby('name')
    res = list()
    orderData = []
    for name, group in mission.groupby('name'):
        # print(np.array(group)[:, [0,1,1]])
        for i in np.array(group)[:, [0,1,2]]:
            orderData.insert(0, list(i))
        # orderData = np.vstack((np.array(group)[:, [0,1,2]], orderData))
        nameList.append(name)
        res.append(list(i[0] for i in np.array(group)[:, [1]]))
    return res, orderData
# print(distList, 'distList'

# print(np.reshape(cal, [1, -1]))
def reduce(data):
    res = str(data)
    #替换掉'['和']'
    res = res.replace('[','')
    res = res.replace(']','')
    #最后转化成列表
    return list(eval(res))
# print(reduce(cal))
def split(dist, data):
    count = 0
    res = list()
    # print(data, dist)
    for item in np.array(dist):
        res.append('4/' + str(data[count: count + item][0]) + '-4/' + str(data[count: count + item][-1]))
        count += item
    return res
if __name__ == "__main__":
    rowdistList = pd.read_excel('./data/data.xlsx', sheet_name='Sheet1')
    nameList = []
    distList = np.array(rowdistList)[:, [0, 1, 2]]
    cal = calendar.monthcalendar(2019, 4)
    for index, item in enumerate(cal):
        if index % 2 == 0:
            cal[index] = list(np.delete(item, [len(item) - 1, len(item) - 2], axis=0))
        else: cal[index] = list(np.array(item))
    write, orderData = formate(distList)
    dateList = reduce(cal)
    finalarrage = np.array([])
    for index, item in enumerate(write):
        finalarrage = np.append(finalarrage, split(item, dateList))
    orderData = np.array(orderData)
    # orderData = np.array(orderData)
    # print(orderData[:, 1
    df1 = pd.DataFrame({"任务": orderData[:, 0], '天数': orderData[:, 1], '名字': orderData[:, 2], '日期':finalarrage})
    df1.to_excel('./data/test.xlsx',sheet_name='test',index=False)
    
    print(All)
