# 原理：HMM TF-IDF 引用JIEBA库 
import jieba
from gensim.models import word2vec
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
import numpy as np;
import pandas as pd;
import xml.dom.minidom as xmldom
import xml.etree.ElementTree as ET 
import re

path = './data/news_sohusite_xml.smarty/news_sohusite_xml.smarty.xml'
# data = pd.read_table(path,header=None,skiprows=[0,1],sep='\s+')
# print(str(data))
tree = ET.parse(path)
root = tree.getroot()
filePath='./data/word.txt'
fileTrainRead = []
with open(filePath,'wb') as fW:
    #  for i in range(len(seg_list)):
    #     fW.write(seg_list[i].encode('utf-8'))
    #     fW.write('\n'.encode('utf-8'))
    for doc in root.findall('doc'):
        text = doc.find('content').text
        content = '<content>' + str(text) + '</content>'
        fW.write(content.encode('utf-8'))
        fW.write('\n'.encode('utf-8'))
        fileTrainRead.append(str(text))

fileSegWordDonePath ='./data/corpusSegDone.txt'
# read the file by line


# define this function to print a list with Chinese
def PrintListChinese(List):
    for i in range(len(List)):
        print(List[i])
# segment word with jieba
fileTrainSeg=[]
for i in range(len(fileTrainRead)):
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][9:-11],cut_all=False)))])
    if i % 100 == 0 :
        print(i)

# to test the segment result
#PrintListChinese(fileTrainSeg[10])

# save the result
print('end')
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))
        fW.write('\n'.encode('utf-8'))

sentences = word2vec.Text8Corpus("./data/corpusSegDone.txt")
model = word2vec.Word2Vec(sentences,size=200)
model.save("./data/corpusSegDone.bin")
texts = [['human', 'interface', 'computer']]
dct = Dictionary(texts)
# test = model.most_similar("联合国", topn=100)
vector = model.wv['象山']
print(vector)
# print(model['阳光工程'])