# 原理：HMM TF-IDF 引用JIEBA库 
import jieba

seg_list = jieba.cut("毫无疑问,“黑曼巴”科比·布莱恩特是NBA历史上最伟大的球员之一,也许在近20年来也是最受欢迎的篮球运动员", cut_all=True)

print(" / ".join(seg_list))