import cv2
import hashlib
import os,sys

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

if not os.path.exists("output"):
    os.mkdir("output")

h = 40
w = 250

path = r"F:\ccpd_dataset\ccpd_base" # 根据自己的实际情况
fi = open("label.txt","w",encoding="utf-8")
for img_name in os.listdir(path):
    
    # 读取图片的完整名字
    image = cv2.imread(path + "/" + img_name)
    
    # 以 - 为分隔符，将图片名切分，其中iname[4]为车牌字符，iname[2]为车牌坐标
    iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    tempName = iname[4].split("_")
    name = provinces[int(tempName[0])] + alphabets[int(tempName[1])] + ads[int(tempName[2])] \
           + ads[int(tempName[3])] + ads[int(tempName[4])] + ads[int(tempName[5])] + ads[int(tempName[6])]
           
    # crop车牌的左上角和右下角坐标
    [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
    
    # crop图片
    img = image[leftUp[1]:rightDown[1],leftUp[0]:rightDown[0]]
    height, width, depth = img.shape
    
    # 要将图片压缩成40*250，计算压缩比
    imgScale = h/height
    deltaD = int((w/imgScale-width)/2)   # (目标宽-实际宽)/2,因为要分别向左、右拓宽，所有需要除以2
    leftUp[0] = leftUp[0] - deltaD       # 切割宽度向左平移，保证补够250
    rightDown[0] = rightDown[0] + deltaD # 切割宽度向右平移，保证补够250

    if(leftUp[0] < 0):                   # 如果向左平移为负，坐标为0
        rightDown[0] = rightDown[0] - leftUp[0]
        leftUp[0] = 0;
    # 按照   高/宽 = 40 / 250 的比例切割,注意切的结果不是40和250
    img = image[leftUp[1]:rightDown[1],leftUp[0]:rightDown[0]]
    # resize成40*250
    newimg = cv2.resize(img,(w,h))

    cv2.imwrite("output/" + img_name, newimg)
    fi.write(img_name + ":" + name +"\r\n")

fi.close()