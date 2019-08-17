import numpy as np;
import math;

test = [3,1,1,2,5,0]

def filter_check(item):
    return item <= len(test) - 1

def getChild(index):
    if index == 0:
        return filter(filter_check, [index + 1, index + 2])
    row = math.ceil(index / 2) - 1
    begin_index = math.pow(2, row) - 1
    end_index = 2 * math.pow(2, row) - 1
    base = (math.ceil((index - begin_index) / 2) - 1) * 4 + ((index + 1 - begin_index) % 2) * 2
    return filter(filter_check, [end_index + base + 2, end_index + base + 3])

def getParent(index):
    if index == 0:
        return 0
    else:
        index += 1
        row = math.floor(math.log2(index))
        begin_index = math.pow(2, row) - 1
        rank = index - begin_index
        return math.pow(2, row - 1) - 1 + math.floor(rank / 4) * 2 + math.ceil((rank % 4) / 2) - 1
def heapDown(list_query):
    child_dict = dict()
    for index, item in enumerate(test):
        print(getParent(index))
heapDown(test)