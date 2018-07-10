import matplotlib.pyplot as plt

import os
import sys
import numpy as np

class Evaluation:
    classNum = 0
    topN = 0
    threshV = 0.0
    className = {}
    classNameN = 0
    def __init__(self,_class_num, _top_n, _thre):
        self.classNum = _class_num
        self.topN = _top_n
        self.threshV = _thre

        self.rightNumC = np.zeros(self.classNum)
        self.wrongNumC = np.zeros(self.classNum) 
        self.rejectNumC = np.zeros(self.classNum)  
        

    def CalEval(self,_cls_vec,_score,_real_cls):
        '''
        _cls_vec and _score: sort from big to small 
        _real_cls: same type with _cls_ve
        '''
        realCls = 0
        if not self.className.has_key(_real_cls):
            self.className[_real_cls] = self.classNameN
            self.classNameN += 1
        
        realCls = self.className[_real_cls]

        if _cls_vec is None:
            self.rejectNumC[realCls] += 1
            return 
        
        if _score[0] < self.threshV:
            self.rejectNumC[realCls] += 1
            return

        findL = 0
        for idx,data in enumerate(_cls_vec):
            if idx >= self.topN:
                break
            if _score[idx] < self.threshV:
                break
            if data == _real_cls:
                findL = 1
                break
        if findL == 1:
            self.rightNumC[realCls] +=1
        else:
            self.wrongNumC[realCls] += 1
        
    def GetResult(self):
        '''
        get precision and recall of class and plot
        '''
        precisionV = [0.0]*(self.classNum)
        recallV = [0.0]*(self.classNum)
        totalRig = 0.0
        totalWro = 0.0
        totalRej = 0.0

        x =[]
        subpre =[]
        subrec =[]
        for key in self.className:
            n = self.className[key]
            precisionV[n] = self.rightNumC[n] /(self.rightNumC[n] + self.wrongNumC[n]+0.001)
            clsNum = self.rightNumC[n] + self.wrongNumC[n]+self.rejectNumC[n]+0.001
            recallV[n] = (self.rightNumC[n]+self.wrongNumC[n]) / clsNum
            print "class %s : precision %f, recall %f, totalNum %d "%(key,precisionV[n],recallV[n],clsNum)
            totalRig += self.rightNumC[n]
            totalWro += self.wrongNumC[n]
            totalRej += self.rejectNumC[n]

            x.append(key)
            subpre.append(precisionV[n])
            subrec.append(recallV[n])
        
        totalPre = totalRig /(totalRig+totalWro)
        totalNum = totalRig + totalWro+totalRej
        totalRec = (totalRig+totalWro)/(totalNum)
        print 'total precision %f, recall %f , totalNum %f '%(totalPre,totalRec,totalNum)
        x.append('all')
        subpre.append(totalPre)
        subrec.append(totalRec)
        
        plt.figure(0)
        plt.xlabel('class')
        plt.ylabel('score')
        titleName = 'goods PR thre=%f top=%d'%(self.threshV,self.topN)
        plt.title(titleName)
        plt.scatter(x,subpre,marker='x',color='g',label='precision',s=30)
        plt.scatter(x,subrec,marker='+',color='r',label='recall',s=30)
        plt.legend(loc='best')
        plt.savefig("./pre-recall-%0.2f-%d.jpg"%(self.threshV,self.topN))
        plt.close(0)