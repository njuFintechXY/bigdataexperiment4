# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:28:06 2018

@author: xiaoyang
"""

import re
pathname = {'positive':1,'negative':2,'neutral':3}

s = ''
for i in pathname.keys():
    for j in range(0,600):
        try:
            with open('files/training_data/'+i+'/'+str(j)+'.txt') as f:
                temp = f.read()
                s+=re.sub('[^\u4E00-\u9FA5]+','',temp)+'\t'+str(pathname[i])+'\n'
        except FileNotFoundError:
            pass

with open('files/combine.txt','w',encoding='utf-8') as w:
    w.write(s)



