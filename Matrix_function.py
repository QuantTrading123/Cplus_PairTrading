# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:53:01 2020

@author: MAI
"""
import numpy as np
from vecm import para_vecm
from scipy.stats import f , chi2

def VAR_model( y , p ):    # p 為落後基數 1-5
    k = len(y.T)     # 幾檔股票 2 
    n = len(y)       # 資料長度
    
    xt = np.ones( ( n-p , (k*p)+1 ) )   #120-p  // 3 5 7 9 11
    
    #print("xt",xt)
    for i in range(n-p):
        a = 1
        for j in range(p):
            a = np.hstack( (a,y[i+p-j-1]) )
        a = a.reshape([1,(k*p)+1])
        xt[i] = a
    
    zt = np.delete(y,np.s_[0:p],axis=0)
    xt = np.mat(xt)
    zt = np.mat(zt)

    beta = ( xt.T * xt ).I * xt.T * zt                      # 計算VAR的參數
    print("beta" ,beta)
    
    A = zt - xt * beta                                      # 計算殘差
    sigma = ( (A.T) * A ) / (n-p)                           # 計算殘差的共變異數矩陣
        
    return [ sigma , beta ]

# 配適 VAR(P) 模型 ，並利用BIC選擇落後期數--------------------------------------------------------------
def order_select( y , max_p ):
    
    k = len(y.T)     # 幾檔股票
    print("幾檔股票",k)
    n = len(y)       # 資料長度
    #print(y)
    bic = np.zeros((max_p,1))
    for p in range(1,max_p+1): # 1-5
        sigma = VAR_model( y , p )[0]
        bic[p-1] = np.log( np.linalg.det(sigma) ) + np.log(n) * p * (k*k) / n
    bic_order = int(np.where(bic == np.min(bic))[0] + 1)        # 因為期數p從1開始，因此需要加1
    print("bic order",bic_order)
    
    return bic_order

