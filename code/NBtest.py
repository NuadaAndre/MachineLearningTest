# -*- coding:utf-8 -*-
#原创：兜哥 https://github.com/duoergun0729/1book

from nltk.probability import FreqDist
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

#测试样本数
N=120

def load_user_cmd_new(filename): #数据搜集和数据清洗
    cmd_list=[]
    dist=[]
    with open(filename) as f:
        i=0
        x=[]
        for line in f:
            line=line.strip('\n')
            x.append(line)
            dist.append(line)
            i+=1
            if i == 100:  #每100个命令组成一个操作序列
                cmd_list.append(x)
                x=[]
                i=0
    fdist = FreqDist(dist).keys()
    return cmd_list,fdist


def get_user_cmd_feature_new(user_cmd_list,dist):
    user_cmd_feature=[]

    for cmd_list in user_cmd_list:
        v=[0]*len(dist)
        for i in range(0,len(dist)):
            if dist[i] in cmd_list:
                v[i]+=1
        user_cmd_feature.append(v)
    return user_cmd_feature

def get_label(filename,index=0):
    x=[]
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            x.append( int(line.split()[index]))
    return x

if __name__ == '__main__':
    user_cmd_list,dist=load_user_cmd_new("../data/MasqueradeDat/User1")
    user_cmd_feature=get_user_cmd_feature_new(user_cmd_list,dist)
    labels=get_label("../data/MasqueradeDat/label.txt",2)
    y=[0]*50+labels

    x_train=user_cmd_feature[0:N]
    y_train=y[0:N]

    x_test=user_cmd_feature[N:150]
    y_test=y[N:150]

    clf = GaussianNB().fit(x_train, y_train)

    y_predict=clf.predict(x_test)
    score = cross_validation.cross_val_score(clf, user_cmd_feature, y, n_jobs=-1, cv=10) #十字交叉验证
    print score
