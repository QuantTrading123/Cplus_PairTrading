# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:47:16 2020

@author: NAVY
"""
from turtle import right
import numpy as np
import numpy.matlib
from scipy.linalg import eigh, eig
from scipy.linalg import orth
import logging


# def JCItestpara(X_data, model_type, lag_p):
#     print("model type :", model_type)
#     if model_type == 'model1':
#         model_type = 1
#     elif model_type == 'model2':
#         model_type = 2
#     elif model_type == 'model3':
#         model_type = 3
#     [NumObs, NumDim] = X_data.shape

#     dY_ALL = X_data[1:, :] - X_data[0:-1, :]
#     dY = dY_ALL[lag_p:, :]  # DY
#     Ys = X_data[lag_p:-1, :]  # Lag_Y

#     # 底下開始處理估計前的截距項與時間趨勢項
#     if lag_p == 0:
#         if model_type == 1:
#             dX = np.zeros([NumObs-1, NumDim])  # DLag_Y
#         elif model_type == 2:
#             dX = np.zeros([NumObs-1, NumDim])  # DLag_Y
#             Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1))))  # Lag_Y
#         elif model_type == 3:
#             dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y
#         elif model_type == 4:
#             dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y
#             Ys = np.hstack(
#                 (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))  # Lag_Y
#         elif model_type == 5:
#             dX = np.hstack((np.ones((NumObs-lag_p-1, 1)), np.arange(1,
#                            NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))

#     elif lag_p > 0:
#         dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p])  # DLag_Y
#         for xi in range(lag_p):
#             dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p -
#                                                           xi - 1:NumObs - xi - 2, :]
#         if model_type == 2:
#             Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1))))
#         elif model_type == 3:
#             dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1))))
#         elif model_type == 4:
#             Ys = np.hstack(
#                 (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))
#             dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1))))
#         elif model_type == 5:
#             dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1)),
#                            np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))

#     # 準備開始估計，先轉成matrix，計算比較直觀
#     dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

#     # 先求dX'*dX 方便下面做inverse
#     dX_2 = dX.T * dX
#     # I-dX * (dX'*dX)^-1 * dX'
#     # python無法計算0矩陣的inverse，用判斷式處理
#     if np.sum(dX_2) == 0:
#         M = np.identity(NumObs-lag_p-1) - dX * dX.T
#     else:
#         M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T

#     R0, R1 = dY.T * M, Ys.T * M

#     S00 = R0 * R0.T / (NumObs-lag_p-1)
#     S01 = R0 * R1.T / (NumObs-lag_p-1)
#     S10 = R1 * R0.T / (NumObs-lag_p-1)
#     S11 = R1 * R1.T / (NumObs-lag_p-1)

#     # 計算廣義特徵值與廣義特徵向量
#     eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)

#     # 排序特徵向量Eig_vector與特徵值lambda
#     sort_ind = np.argsort(-eigValue_lambda)
#     #eigValue_lambda = eigValue_lambda[sort_ind]
#     eigVecs = eigvecs[:, sort_ind]
#     #eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)

#     # Beta
#     jci_beta = np.mat(eigVecs[:, 0][0:2])
#     jci_beta = jci_beta.T
#     # Alpha
#     a = np.mat(eigVecs[:, 0])
#     jci_alpha = S01 * a.T
#     # 初始化 c0, d0, c1, d1
#     c0, d0 = 0, 0
#     c1, d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])

#     # 計算 c0, d0, c1, d1，與殘差及VEC項的前置
#     if model_type == 1:
#         W = dY - Ys * jci_beta * jci_alpha.T
#         P = dX.I * W  # [B1,...,Bq]
#         P = P.T

#     elif model_type == 2:
#         c0 = eigVecs[-1, 0:1]
#         W = dY - (Ys[:, 0:2] * jci_beta +
#                   numpy.matlib.repmat(c0, NumObs-lag_p-1, 1)) * jci_alpha.T
#         P = dX.I * W  # [B1,...,Bq]
#         P = P.T

#     elif model_type == 3:
#         W = dY - Ys * jci_beta * jci_alpha.T
#         P = dX.I * W
#         P = P.T
#         c = P[:, -1]
#         c0 = jci_alpha.I * c
#         c1 = c - jci_alpha * c0

#     elif model_type == 4:
#         d0 = eigVecs[-1, 0:1]
#         W = dY - (Ys[:, 0:2] * jci_beta + np.arange(1, NumObs-lag_p,
#                   1).reshape(NumObs-lag_p-1, 1) * d0) * jci_alpha.T
#         P = dX.I * W
#         P = P.T
#         c = P[:, -1]
#         c0 = jci_alpha.I * c
#         c1 = c - jci_alpha * c0

#     elif model_type == 5:
#         W = dY - Ys * jci_beta * jci_alpha.T
#         P = dX.I * W  # [B1,...,Bq]
#         P = P.T
#         c = P[:, -2]
#         c0 = jci_alpha.I * c
#         c1 = c - jci_alpha * c0
#         d = P[:, -1]
#         d0 = jci_alpha.I * d
#         d1 = d - jci_alpha * d0

#     # 計算殘差
#     ut = W - dX * P.T

#     # 計算VEC項
#     gamma = []
#     for bi in range(1, lag_p+1):
#         Bq = P[:, (bi-1)*NumDim: bi * NumDim]
#         gamma.append(Bq)

#     # 把Ct統整在一起
#     Ct = jci_alpha*c0 + c1 + jci_alpha*d0 + d1

#     return jci_alpha, jci_beta, Ct, ut, gamma


def JCItestpara_spilCt(X_data, model_type, lag_p):
    print("model type & lag p 基數 :", model_type, lag_p)
    [NumObs, NumDim] = X_data.shape
    print("Numobs {}, NumDim {}".format(NumObs, NumDim))

    dY_ALL = X_data[1:, :] - X_data[0:-1, :]
    print("dyALL", dY_ALL)
    dY = dY_ALL[lag_p:, :]  # DY
    Ys = X_data[lag_p:-1, :]  # Lag_Y

    # 底下開始處理估計前的截距項與時間趨勢項
    if lag_p == 0:
        if model_type == 1:
            dX = np.zeros([NumObs-1, NumDim])  # DLag_Y
        elif model_type == 2:
            dX = np.zeros([NumObs-1, NumDim])  # DLag_Y
            Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1))))  # Lag_Y
        elif model_type == 3:
            dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y
        elif model_type == 4:
            dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y
            Ys = np.hstack(
                (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))  # Lag_Y
        elif model_type == 5:
            dX = np.hstack((np.ones((NumObs-lag_p-1, 1)), np.arange(1,
                           NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))

    elif lag_p > 0:
        dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p])  # DLag_Y
        for xi in range(lag_p):
            dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p -
                                                          xi - 1:NumObs - xi - 2, :]
        if model_type == 2:
            Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1))))
        elif model_type == 3:
            dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1))))
        elif model_type == 4:
            Ys = np.hstack(
                (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))
            dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1))))
        elif model_type == 5:
            dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1)),
                           np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))

    # 準備開始估計，先轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)
    print("dX :\n", dX)
    print("dY :\n", dY)
    print("Ys :\n", Ys)
    # 先求dX'*dX 方便下面做inverse
    dX_2 = dX.T * dX
    print('DX2', dX_2)
    # I-dX * (dX'*dX)^-1 * dX'
    # python無法計算0矩陣的inverse，用判斷式處理
    if np.sum(dX_2) == 0:
        M = np.identity(NumObs-lag_p-1) - dX * dX.T
    else:
        M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T

    R0, R1 = dY.T * M, Ys.T * M

    S00 = R0 * R0.T / (NumObs-lag_p-1)
    S01 = R0 * R1.T / (NumObs-lag_p-1)
    S10 = R1 * R0.T / (NumObs-lag_p-1)
    S11 = R1 * R1.T / (NumObs-lag_p-1)
    print("S00 :\n", S00)
    print("S01 :\n", S01)
    print("S10 :\n", S10)
    print("S11 :\n", S11)
    print("A :\m", S10 * S00.I * S01)
    # 計算廣義特徵值與廣義特徵向量
    eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)

    print("eigValue_lambda :\n", eigValue_lambda)
    print("eigvector :\n", eigvecs)
    numpy_eigval, numpy_eigvec = np.linalg.eigh(S11.I @ (S10 @ S00.I @ S01))
    eig_val, vl = eig(a=S10 * S00.I * S01, b=S11, left=True, right=False)
    print("scipy eig val : {}".format(eig_val))
    print("scipy eigh vl ", vl)
    print("numpy val :", numpy_eigval)

    # 排序特徵向量Eig_vector與特徵值lambda
    print("eigenvalue of A \n", S10 @ S00.I @ S01)
    print("dot of eigenvalue & eigenvector 1\n ", S10 @ S00.I @ S01 @ eigvecs)
    print("dot of eigenvalue & eigenvector 2\n ",
          eigValue_lambda[0] * S11 @ eigvecs)
    sort_ind = np.argsort(-eigValue_lambda)
    print("sort index :", sort_ind)
    # eigValue_lambda = eigValue_lambda[sort_ind]
    print("eigvecs :", eigvecs)
    eigVecs = eigvecs[:, sort_ind]
    print("eigVecs :\n", eigVecs)
    # vl = vl[:, sort_ind]
    # 將所有eigenvector同除第一行的總和
    print("第一行總和", np.sum(np.absolute(eigVecs[:, 0][0:2])))
    eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:, 0][0:2]))
    vl = vl/np.sum(np.absolute(vl[:, 0][0:2]))
    print("eigvecs st :\n", eigVecs_st)
    print("vl :", vl)
    # eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    print(eigVecs_st[:, 0])
    # Beta
    jci_beta = eigVecs_st[:, 0][0:2].reshape(NumDim, 1)
    print("jci beta", jci_beta)
    # Alpha
    a = np.mat(eigVecs_st[:, 0])
    print("a :", a)
    jci_a = S01 * a.T
    print("jci_a", jci_a)
    jci_alpha = jci_a/np.sum(np.absolute(jci_a))
    print("jci alpha", jci_alpha)

    # 初始化 c0, d0, c1, d1
    c0, d0 = 0, 0
    c1, d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])

    # 計算 c0, d0, c1, d1，與殘差及VEC項的前置
    if model_type == 1:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T

    elif model_type == 2:
        c0 = eigVecs_st[-1, 0:1]
        W = dY - (Ys[:, 0:2] * jci_beta +
                  numpy.matlib.repmat(c0, NumObs-lag_p-1, 1)) * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T

    elif model_type == 3:
        W = dY - Ys * jci_beta * jci_alpha.T
        print("W :", W)
        print("dX :", dX.I)
        print("numpy of dx i", np.linalg.pinv(dX))
        P = dX.I * W
        P = P.T
        c = P[:, -1]
        print("P :", P)
        print("c :", c)
        c0 = jci_alpha.I * c
        print("C0 :", c0)
        c1 = c - jci_alpha * c0
        print("c1 :", c1)

    elif model_type == 4:
        d0 = eigVecs_st[-1, 0:1]
        W = dY - (Ys[:, 0:2] * jci_beta + np.arange(1, NumObs-lag_p,
                  1).reshape(NumObs-lag_p-1, 1) * d0) * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:, -1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0

    elif model_type == 5:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        c = P[:, -2]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        d = P[:, -1]
        d0 = jci_alpha.I * d
        d1 = d - jci_alpha * d0
    # 計算殘差
    ut = W - dX * P.T
    Ct_all = jci_alpha*c0 + c1 + jci_alpha*d0 + d1

    # 計算VEC項
    gamma = []
    for bi in range(1, lag_p+1):
        Bq = P[:, (bi-1)*NumDim: bi * NumDim]
        gamma.append(Bq)
    temp1 = np.dot(np.dot(jci_beta.transpose(), S11[0:2, 0:2]), jci_beta)
    omega_hat = S00[0:2, 0:2] - \
        np.dot(np.dot(jci_alpha, temp1), jci_alpha.transpose())
    # 把Ct統整在一起
    print("tmp111 :", temp1)
    print("omega_hat", omega_hat)
    print("d0", d0)
    print("d1", d1)
    print('ut :', ut)
    print('Ct_all', Ct_all)
    Ct = []
    Ct.append(c0)
    Ct.append(d0)
    Ct.append(c1)
    Ct.append(d1)
    print("JCITEST ==============================")
    print(jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat)
    print("JCITEST ==============================")
    return jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat


def JCItest_withTrace(X_data, model_type, lag_p):  # 2黨數據 , 模型type , lag基數
    # trace test
    print('==============================================')
    print("model type : ", model_type)
    print("lag_p :", lag_p)
    [NumObs, NumDim] = X_data.shape  # (120,2)
    # print("X_Data first",X_data[1:, :])
    # print("X_Data second",X_data[0:-1, :])
    dY_ALL = X_data[1:, :] - X_data[0:-1, :]  # 算截距
    # print("dY_ALL :",dY_ALL)
    # print(len(dY_ALL))
    dY = dY_ALL[lag_p:, :]  # DY
    Ys = X_data[lag_p:-1, :]  # Lag_Y

    # 底下開始處理估計前的截距項與時間趨勢項
    if lag_p == 0:
        if model_type == 1:
            dX = np.zeros([NumObs-1, NumDim])  # DLag_Y
            # print("model type 1 dX :" ,dX)
        elif model_type == 2:
            dX = np.zeros([NumObs-1, NumDim])  # DLag_Y
            Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1))))  # Lag_Y
            # print("model type 2 Ys :" ,Ys)
            # print("model type 2 dX :" ,dX)
        elif model_type == 3:
            dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y
            # print("model type 3 dX :" ,dX)
        elif model_type == 4:
            dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y
            Ys = np.hstack(
                (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))  # Lag_Y
            # print("model type 4 Ys :" ,Ys)
            # print("model type 4 dX :" ,dX)
        elif model_type == 5:
            dX = np.hstack((np.ones((NumObs-lag_p-1, 1)), np.arange(1,
                           NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))
            # print("model type 5 dX :" ,dX)
    elif lag_p > 0:
        dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p])  # DLag_Y
        for xi in range(lag_p):
            dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p -
                                                          xi - 1:NumObs - xi - 2, :]
        if model_type == 2:
            Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1))))
        elif model_type == 3:
            dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1))))
            # print("dX lagp",dX)
        elif model_type == 4:
            Ys = np.hstack(
                (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))
            dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1))))
        elif model_type == 5:
            dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1)),
                           np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))

    # 準備開始估計，先轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)
    # 先求dX'*dX 方便下面做inverse
    # print("dx shape", dX.shape)
    # print("dY shape", dY.shape)
    # print("Ys shape", Ys.shape)

    dX_2 = dX.T * dX
    # print("DX22222",dX_2)
    # I-dX * (dX'*dX)^-1 * dX'
    # python無法計算0矩陣的inverse，用判斷式處理
    if np.sum(dX_2) == 0:
        M = np.identity(NumObs-lag_p-1) - dX * dX.T
    else:
        M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T
        # print("MMMM",dX_2.I)

    R0, R1 = dY.T * M, Ys.T * M

    S00 = R0 * R0.T / (NumObs-lag_p-1)
    S01 = R0 * R1.T / (NumObs-lag_p-1)
    S10 = R1 * R0.T / (NumObs-lag_p-1)
    S11 = R1 * R1.T / (NumObs-lag_p-1)

    # 計算廣義特徵值與廣義特徵向量
    # print("A",S10 * S00.I * S01)
    # print("B",S11)

    eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    print("eigvalue lambda :\n", eigValue_lambda)
    print("eigvector :\n", eigvecs)
    # 排序特徵向量Eig_vector與特徵值lambda
    sort_ind = np.argsort(-eigValue_lambda)
    eigValue_lambda = eigValue_lambda[sort_ind]

    eigVecs = eigvecs[:, sort_ind]
    # 將所有eigenvector同除第一行的總和
    # eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:,0][0:2]))

    eigValue_lambda = eigValue_lambda.reshape(len(eigValue_lambda), 1)

    # Beta
    # jci_beta = eigVecs_st[:,0][0:2].reshape(NumDim,1)
    jci_beta = eigVecs[:, 0][0:2].reshape(NumDim, 1)

    '''
    # Alpha
    a = np.mat(eigVecs_st[:,0])
    jci_a = S01 * a.T
    jci_alpha = jci_a/np.sum(np.absolute(jci_a))
    '''
    # Alpha
    a = np.mat(eigVecs[:, 0])
    print('a \n', a)
    jci_alpha = S01 * a.T

    # 初始化 c0, d0, c1, d1
    c0, d0 = 0, 0
    c1, d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])

    # 計算 c0, d0, c1, d1，與殘差及VEC項的前置
    if model_type == 1:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        cvalue = [12.3329, 4.1475]
    elif model_type == 2:
        # c0 = eigVecs_st[-1, 0:1]
        c0 = eigVecs[-1, 0:1]
        print("remat :", numpy.matlib.repmat(c0, NumObs-lag_p-1, 1))
        W = dY - (Ys[:, 0:2] * jci_beta +
                  numpy.matlib.repmat(c0, NumObs-lag_p-1, 1)) * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        cvalue = [20.3032, 9.1465]
    elif model_type == 3:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:, -1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        cvalue = [15.4904, 3.8509]
    elif model_type == 4:
        # d0 = eigVecs_st[-1, 0:1]
        d0 = eigVecs[-1, 0:1]
        W = dY - (Ys[:, 0:2] * jci_beta + np.arange(1, NumObs-lag_p,
                  1).reshape(NumObs-lag_p-1, 1) * d0) * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:, -1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        cvalue = [25.8863, 12.5142]
    elif model_type == 5:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        c = P[:, -2]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        d = P[:, -1]
        d0 = jci_alpha.I * d
        d1 = d - jci_alpha * d0
        cvalue = [18.3837, 3.8395]
    # 計算殘差
    ut = W - dX * P.T
    Ct_all = jci_alpha*c0 + c1 + jci_alpha*d0 + d1

    # 計算VEC項
    gamma = []
    for bi in range(1, lag_p+1):
        Bq = P[:, (bi-1)*NumDim: bi * NumDim]
        gamma.append(Bq)
    temp1 = np.dot(np.dot(jci_beta.transpose(), S11[0:2, 0:2]), jci_beta)
    omega_hat = S00[0:2, 0:2] - \
        np.dot(np.dot(jci_alpha, temp1), jci_alpha.transpose())
    # 把Ct統整在一起
    Ct = []
    Ct.append(c0)
    Ct.append(d0)
    Ct.append(c1)
    Ct.append(d1)

    TraceTest_H = []
    TraceTest_T = []
    print("eig_labd", 1-eigValue_lambda)
    for rn in range(0, NumDim):
        print('eeel :', 1-eigValue_lambda[rn:NumDim, :])
        eig_lambda = np.cumprod(1-eigValue_lambda[rn:NumDim, :])
        print("eig_lambda : \n", eig_lambda[-1])
        trace_stat = -2 * np.log(eig_lambda[-1] ** ((NumObs-lag_p-1)/2))
        TraceTest_H.append(cvalue[rn] < trace_stat)
        TraceTest_T.append(trace_stat)
    #print("eig_lambda : \n", eig_lambda)
    print("---------------------")
    print("TraceTest_H : \n", TraceTest_H)
    print("---------------------")
    print("TraceTest_T : \n", TraceTest_T)
    print("---------------------")
    # print("JCITRACE ==============================")
    # print(jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat)
    # print("JCITRACE ==============================")
    return TraceTest_H, TraceTest_T, jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat


def JCI_AutoSelection(Row_Y, opt_q):
    # 論文中的BIC model selection
    [NumObs, k] = Row_Y.shape
    # print("NumObs :", NumObs)
    # print("k in JCI_AutoSelection :",k)
    opt_p = opt_q + 1
    Tl = NumObs - opt_p

    TraceTest_table = np.zeros([5, k])
    # print("TraceTest_table :",TraceTest_table)
    BIC_table = np.zeros([5, 1])
    BIC_List = np.ones([5, 1]) * np.Inf
    opt_model_num = 0
    for mr in range(0, 5):  # production is 5
        tr_H, _, _, _, _, ut, _, _, _ = JCItest_withTrace(Row_Y, mr+1, opt_q)
        # print("-------------------------------------")
        # print("MR :", mr)
        # print("tr_H :", tr_H)
        # print("ut : ", ut)
        # print("-------------------------------------")
        # 把結果存起來，True是拒絕，False是不拒絕，tr_H[0]是Rank0,tr_H[1]是Rank1
        TraceTest_table[mr, :] = tr_H
        # print("Trace Test table : ", TraceTest_table)
        # 以下計算BIC，僅計算Rank1
        eps = np.mat(ut)
        sq_Res_r1 = eps.T * eps / Tl
        errorRes_r1 = eps * sq_Res_r1.I * eps.T
        trRes_r1 = np.trace(errorRes_r1)
        L = (-k*Tl*0.5)*np.log(2*np.pi) - (Tl*0.5) * \
            np.log(np.linalg.det(sq_Res_r1)) - 0.5*trRes_r1

        if mr == 0:
            # alpha(k,1) + beta(k,1) + q*Gamma(k,k)
            deg_Fred = 2*k + opt_q*(k*k)
        elif mr == 1:
            # alpha(k,1) + beta(k,1) + C0(1,1) + q*Gamma(k,k)
            deg_Fred = 2*k + 1 + opt_q*(k*k)
        elif mr == 2:
            # alpha(k,1) + beta(k,1) + C0(1,1) + C1(k,1) + q*Gamma(k,k)
            deg_Fred = 3*k + 1 + opt_q*(k*k)
        elif mr == 3:
            # alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + q*Gamma(k,k)
            deg_Fred = 3*k + 2 + opt_q*(k*k)
        elif mr == 4:
            # alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + D1(k,1) + q*Gamma(k,k)
            deg_Fred = 4*k + 2 + opt_q*(k*k)
        # 把Rank1各模型的BIC存起來
        BIC_table[mr] = -2*np.log(L) + deg_Fred*np.log(NumObs*k)

        # 挑出被選的Rank1模型
        if TraceTest_table[mr, 0] == 1 and TraceTest_table[mr, 1] == 0:
            # 拒絕R0，不拒絕R1，該模型的最適Rank為R1，並把該模型與Rank1的BIC值存起來
            BIC_List[mr] = BIC_table[mr]
            opt_model_num += 1
        # elif TraceTest_table[mr, 0] == 0 and TraceTest_table[mr, 1] == 0:
        #     # 不拒絕R0，那R1應該是不用測，該模型的最適Rank為R0，紀錄為NaN
        #     continue
        # elif TraceTest_table[mr, 0] == 0 and TraceTest_table[mr, 1] == 1:
        #     # 不拒絕R0，那R1應該是不用測，該模型的最適Rank為R0，紀錄為NaN
        #     continue
        # elif TraceTest_table[mr, 0] == 1 and TraceTest_table[mr, 1] == 1:
        #     # 拒絕R0且拒絕R1，該模型的最適Rank為R2，紀錄為NaN
        #     continue

    BIC_List = BIC_List.tolist()
    # 找出有紀錄的BIC中最小值，即為Opt_model，且Opt_model+1就對應我們的模型編號
    Opt_model = BIC_List.index(min(BIC_List))

    if opt_model_num == 0:
        # 如果opt_model_num是0，代表沒有最適模型或最適模型為Rank0
        return 0
    else:
        # 如果opt_model_num不是0，則Opt_model+1模型的Rank1即為我們最適模型
        return Opt_model+1


def Johansen_mean(alpha, beta, gamma, mu, lagp, NumDim=2):
    # 論文中的closed form mean
    # lagp指的是VECM的LAG期數
    print("alpha :", alpha)
    print("beta :", beta)
    print("gamma :", gamma)
    print("mu : ", mu)
    print("mu ", len(mu), len(mu[0]))
    sumgamma = np.zeros([NumDim, NumDim])
    print(len(gamma))
    for i in range(0, lagp):
        sumgamma = sumgamma+gamma[i]
    print("sumgamma :", sumgamma)
    GAMMA = np.eye(NumDim) - sumgamma
    print("GAMMA :", GAMMA)
    # 計算正交化的alpha,beta
    alpha_orthogonal = alpha.copy()
    alpha_t = alpha.transpose()
    alpha_orthogonal[1, 0] = (-(alpha_t[0, 0] *
                                alpha_orthogonal[0, 0])) / alpha_t[0, 1]
    print("alpha orthogonal :", alpha_orthogonal)
    print("sum of orthogonal", sum(abs(alpha_orthogonal)))
    alpha_orthogonal = alpha_orthogonal/sum(abs(alpha_orthogonal))
    beta_orthogonal = beta.copy()
    beta_t = beta.transpose()
    beta_orthogonal[1, 0] = - \
        ((beta_t[0, 0]*beta_orthogonal[0, 0])) / beta_t[0, 1]
    print("beta_orthogonal", beta_orthogonal)
    beta_orthogonal = beta_orthogonal/sum(abs(beta_orthogonal))
    print("beta_orthogonal", beta_orthogonal)
    # 計算MEAN
    temp1 = np.linalg.inv(
        np.dot(np.dot(alpha_orthogonal.transpose(), GAMMA), beta_orthogonal))
    print("tmp1", temp1)
    C = np.dot(np.dot(beta_orthogonal, temp1), alpha_orthogonal.transpose())
    temp2 = np.linalg.inv(np.dot(alpha.transpose(), alpha))
    alpha_hat = np.dot(alpha, temp2)
    temp3 = np.dot(GAMMA, C) - np.eye(NumDim)
    print("temp3 :", temp3)
    print("alpha", alpha)
    C0 = np.mat(mu[0])
    C1 = np.mat(mu[2])
    D0 = np.mat(mu[1])
    D1 = np.mat(mu[3])
    print('C0 : {}, C1 : {}, D0 :{}, D1 :{} '. format(C0, C1, D0, D1))
    C0 = alpha*C0 + C1 + alpha*D0 + D1
    Ct = alpha*D0 + D1
    expect_intcept = np.dot(np.dot(alpha_hat.transpose(), temp3), C0)
    expect_slope = np.dot(np.dot(alpha_hat.transpose(), temp3), Ct)
    return expect_intcept, expect_slope


def Johansen_std_correct(alpha, beta, ut, mod_gamma, lag_p, rank=1):
    # print("================= Johanson std correct ===================")
    # print("alpha :", alpha)
    # print("beta :", beta)
    # print("ut", ut)
    # print("mod gamma", mod_gamma)
    # print("lagged P", lag_p)
    # 論文中的closed form std
    NumDim = 2
    if lag_p > 0:
        # 建立～A
        tilde_A_11 = alpha
        tilde_A_21 = np.zeros([NumDim*lag_p, 1])
        tilde_A_12 = np.zeros([NumDim, NumDim*lag_p])

        # 建立～B
        tilde_B_11 = beta
        # tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        tilde_B_3 = np.zeros([NumDim + NumDim*lag_p, NumDim*lag_p])

        # 用同一個迴圈同時處理～A與～B
        for qi in range(lag_p):
            tilde_A_12[0:NumDim, qi*NumDim:(qi+1)*NumDim] = mod_gamma[qi]
            tilde_B_3[qi*NumDim:NumDim*(2+qi), qi*NumDim:(qi+1) *
                      NumDim] = np.vstack([np.eye(NumDim), -np.eye(NumDim)])
        # print("tilde_A_12", tilde_A_12)
        # print("tilde B 3", tilde_B_3)
        tilde_A_22 = np.eye(NumDim*lag_p)
        tilde_A = np.hstack(
            [np.vstack([tilde_A_11, tilde_A_21]),  np.vstack([tilde_A_12, tilde_A_22])])
        tilde_B = np.hstack([np.vstack([tilde_B_11, tilde_A_21]), tilde_B_3])
        # print("tilde_A : ", tilde_A)
        # print("tilde_B : ", tilde_B)

    elif lag_p == 0:
        tilde_A = alpha
        tilde_B = beta
    tilde_Sigma = np.zeros([NumDim*(lag_p+1), NumDim*(lag_p+1)])
    tilde_Sigma[0:NumDim, 0:NumDim] = np.dot(ut.transpose(), ut)/(len(ut)-1)
    # print("UTUTUTUUTUTUTUTUT,", np.dot(ut.transpose(), ut))
    # print("tilde sigma", tilde_Sigma)
    # print("length of ut ", len(ut))
    tilde_J = np.zeros([1, 1+NumDim*(lag_p)])
    tilde_J[0, 0] = 1
    if lag_p == 0:
        temp1 = np.eye(rank)+np.dot(beta.transpose(), alpha)
        temp2 = np.kron(temp1, temp1)
        temp3 = np.linalg.inv(np.eye(rank)-temp2)
        omega = np.dot(ut.transpose(), ut)/(len(ut)-1)
        temp4 = np.dot(np.dot(beta.transpose(), omega), beta)
        var = np.dot(temp3, temp4)
        print("var", var)
    else:
        temp1 = np.eye(NumDim*(lag_p+1)-1)+np.dot(tilde_B.transpose(), tilde_A)
        temp2 = np.kron(temp1, temp1)
        k = (NumDim*(lag_p+1)-1)*(NumDim*(lag_p+1)-1)
        temp3 = np.linalg.inv(np.eye(k)-temp2)
        # print("tmp3", temp3)
        # print("tilde sigma", tilde_Sigma)
        temp4 = np.dot(np.dot(tilde_B.transpose(), tilde_Sigma), tilde_B)
        # print("tmp4 : ", temp4)
        temp4 = temp4.flatten('F')
        # print("tmp4 : ", temp4)
        temp5 = np.dot(temp3, temp4)
        # print("tmp5 :", temp5)
        sigma_telta_beta = np.zeros([NumDim*(lag_p+1)-1, NumDim*(lag_p+1)-1])
        for i in range(NumDim*(lag_p+1)-1):
            for j in range(NumDim*(lag_p+1)-1):
                sigma_telta_beta[i][j] = temp5[0, i+j*(NumDim*(lag_p+1)-1)]
        # print("sigma teladabeta  :", sigma_telta_beta)
        var = np.dot(np.dot(tilde_J, sigma_telta_beta), tilde_J.transpose())
        print("var : ", var)
    return var
