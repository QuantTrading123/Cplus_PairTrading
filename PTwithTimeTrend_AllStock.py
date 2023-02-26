# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:27:14 2020

@author: Hua
"""
import numpy as np
import pandas as pd
import mt
from Matrix_function import order_select
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import JCItestpara_20201113 as jci


def get_key(dict, value):
    tmp = [k for k, v in dict.items() if v == value]
    # print(tmp[0])
    return tmp[0]


def refactor_formation_table(Smin):
    rowAS = np.log(Smin)  # (120, 2)
    B = np.zeros([2, 1])  # B為共整合係數
    CapitW = np.zeros([2, 1])  # CW為資金權重Capital Weight

    # 配適 VAR(P) 模型 ，並利用BIC選擇落後期數，max_p意味著會檢查2~max_p
    try:
        max_p = 5
        p = order_select(rowAS, max_p)  # 選擇最適落後其數
        # if p >= 2:
        #     print("find 2 la")
        #     return [0]
        # ADF TEST
        if p < 1:
            return []
            # adf test
            # portmanteau test

        # var model test whitenoise
        model = VAR(rowAS)
        print("whitenoise p-value", model.fit(p).test_whiteness(nlags=5).pvalue)
        if model.fit(p).test_whiteness(nlags=5).pvalue < 0.05:
            return []
        # Normality test 目前沒用到 不然會有很多pair被del
        '''
        if model.fit(p).test_normality().pvalue < 0.05:
            return []
'           '''
        opt_model = jci.JCI_AutoSelection(
            rowAS, p-1)  # bic based model selection
        #print("model select :", opt_model)
        # 如果有共整合，紀錄下Model與opt_q
        print("Row AS :", rowAS)
        print("opt model", opt_model)
        print("p", p)

        F_a, F_b, F_ct, F_ut, F_gam, ct, omega_hat = jci.JCItestpara_spilCt(
            rowAS, opt_model, p-1)
        Com_para = []
        Com_para.append(F_a)
        Com_para.append(F_b)
        Com_para.extend(F_ct)
        print("Com_parameters : ", Com_para)
        # 把  arrary.shape(2,1) 的數字放進 shape(2,) 的Serires
        # 取出共整合係數
        B[:, 0] = pd.DataFrame(F_b).stack()
        print("BBBBBBB", B[:, 0])
        # 將共整合係數標準化，此為資金權重Capital Weight
        CapitW[:, 0] = B[:, 0] / np.sum(np.absolute(B[:, 0]))

        # 計算Spread的時間趨勢均值與標準差 model 1-5
        print("in mean")
        Johansen_intcept, Johansen_slope = jci.Johansen_mean(
            F_a, F_b, F_gam, F_ct, p-1)
        print("in std")
        Johansen_var_correct = jci.Johansen_std_correct(
            F_a, F_b, F_ut, F_gam, p-1)
        Johansen_std = np.sqrt(Johansen_var_correct)
        print("Johansen_intcept :", Johansen_intcept)
        Johansen_intcept = Johansen_intcept[0, 0]
        Johansen_std = Johansen_std[0, 0],

        return [Johansen_intcept, Johansen_std, opt_model, CapitW[0, 0], CapitW[1, 0]]

    except:
        return []
