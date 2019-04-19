import sys
sys.path.append("/Users/kaixinzeng/PycharmProjects/xinyan/HAIER/")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from ScoreCardFunc import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

global missing_value
missing_value = [-1,-99998,-99999,-999977,-999976,-999978,-999979,-99992]

def MissingRatioSelect(Data,ratio):
    features = list(Data.columns)
    features.remove('target')
    y = {}
    for col in features:
        y[col] = Data[col].apply(lambda x: x >= 0)
    y = pd.DataFrame(y).apply(sum) / 39996
    y = y.reset_index()
    return  list(y[y[0]>ratio]['index'])


def RoughSelect(Data,iv_threshold):
    var_IV = {}
    var_WOE = {}
    var_cutoff = {}
    features = list(Data.columns)
    features.remove('target')
    for col in features:
        if len(Data[col].unique()) == 1:
            continue
        col_Bin = col+'_Bin'
        Data[col_Bin],var_cutoff[col] = pd.qcut(Data[col],20,duplicates='drop',retbins= True)
        cal_woe = CalcWOE(Data,col_Bin,'target')
        var_WOE[col] = cal_woe['WOE']
        var_IV[col] = cal_woe['IV']
        print('Roughly var {} IV is {}'.format(col,var_IV[col]))
    varByIV = [k for k in var_IV if var_IV[k] > iv_threshold]
    print("After Roughly Selection, {} vars left.".format(len(varByIV)))
    return varByIV

def CarefulSelect(Data,iv_threshold):
    var_IV = {}
    var_WOE = {}
    var_cutoff = {}
    features = list(Data.columns)
    features.remove('target')
    cnt = 0
    for col in features:
        cnt += 1
        print("{}/{}:{} is in processing".format(cnt,len(features),col))
        col1 = str(col) + '_Bin'
        special_attribute = [k for k in set(missing_value) if k in set(Data[col])]
        cutOffPoints = ChiMerge_MaxInterval(Data,col,'target',special_attribute = special_attribute)
        var_cutoff[col] = cutOffPoints
        Data[col1] = Data[col].map(lambda x:AssignBin(x,cutOffPoints,special_attribute = special_attribute))

        BRM = BadRateMonotone(Data,col1,'target',special_attribute = special_attribute)
        if not BRM:
            for bins in range(9,len(missing_value),-1):
                cutOffPoints = ChiMerge_MaxInterval(Data,col,'target',max_interval = bins,special_attribute = special_attribute)
                Data[col1] = Data[col].map(lambda x:AssignBin(x,cutOffPoints,special_attribute = special_attribute))
                BRM = BadRateMonotone(Data, col1, 'target', special_attribute=special_attribute)
                if BRM:
                    break
            var_cutoff[col] = cutOffPoints

        maxPcnt = MaximumBinPcnt(Data,col1)
        if maxPcnt > 0.9:
            print("     delete var {} because the maximun bin occupies more than 90%".format(col))
            continue
        cal_woe = CalcWOE(Data, col1, 'target')
        var_IV[col] = cal_woe['IV']
        var_WOE[col] = cal_woe['WOE']
        # print(cal_woe['WOE'])
    varByIV = [k for k in var_IV.items() if k[1] > iv_threshold]
    varByIV = dict(varByIV)
    print("After Careful Selection, {} vars left.".format(len(varByIV)))
    return var_cutoff,varByIV,var_WOE

def WOE_trains(Data,Params):
    for col in Params[1].keys():
        col2 = str(col) + "_WOE"
        if col in Params[0].keys():
            cutOfPoints = Params[0][col]
            special_attribute = [k for k in set(missing_value) if k in set(Data[col])]
            binValue = Data[col].map(lambda x: AssignBin(x,cutOfPoints,special_attribute = special_attribute))
            try:
                Data[col2] = binValue.map(lambda x: Params[2][col][x])
            except KeyError:
                print(col,KeyError)
        else:
            Data[col2] = Data[col].map(lambda x: Params[2][col][x])
    lists = [str(k)+'_WOE' for k in Params[1]]
    return Data[lists+['target']]

def CorrSelect(Data,IV_Var,roh_thresould):
    removedVar = []
    cols = list(Data.columns)
    cols.remove('target')
    m = np.matrix(Data[cols])
    pearsonCorr = np.corrcoef(m)
    for i in range(len(cols)-1):
        for j in range(i+1,len(cols)):
            if abs(pearsonCorr[i,j]) >= roh_thresould \
                and cols[i] not in removedVar \
                and cols[j] not in removedVar:
                print("Var {0} and Var {1} Corr is {2}".format(cols[i].replace("_WOE",""),cols[j].replace("_WOE",""), abs(pearsonCorr[i, j])))
                if IV_Var[cols[i].replace("_WOE","")] > IV_Var[cols[j].replace("_WOE","")]:
                    removedVar.append(cols[j])
                else:
                    removedVar.append(cols[i])
    cols = [k for k in cols if k not in removedVar]
    return cols

def CorrSelect2(Data, IV_Var, roh_thresould):
    removedVar = []
    cols = list(Data.columns)
    cols.remove('target')
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            roh = np.corrcoef([Data[cols[i]], Data[cols[j]]])[0, 1]
            if abs(roh) >= roh_thresould \
                    and cols[i] not in removedVar \
                    and cols[j] not in removedVar:
                print("Var {0} and Var {1} Corr is {2}".format(cols[i].replace("_WOE", ""),
                                                               cols[j].replace("_WOE", ""), abs(roh)))
                if IV_Var[cols[i].replace("_WOE", "")] > IV_Var[cols[j].replace("_WOE", "")]:
                    removedVar.append(cols[j])
                else:
                    removedVar.append(cols[i])
    cols = [k for k in cols if k not in removedVar]
    return cols


    # for i in range(len(var_IV_sorted) - 1):
    #     if var_IV_sorted[i] not in removed_var:
    #         x1 = var_IV_sorted[i] + "_WOE"
    #         for j in range(i + 1, len(var_IV_sorted)):
    #             if var_IV_sorted[j] not in removed_var:
    #                 x2 = var_IV_sorted[j] + "_WOE"
    #                 roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
    #                 if abs(roh) >= roh_thresould:
    #                     print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
    #                     if var_IV[var_IV_sorted[i]] > var_IV[var_IV_sorted[j]]:
    #                         removed_var.append(var_IV_sorted[j])
    #                     else:
    #                         removed_var.append(var_IV_sorted[i])
    # var_IV_sortet_2 = [i for i in var_IV_sorted if i not in removed_var]


def VIFSelect(Data,IV_Var,vif_thresould = 10):
    cols = list(Data.columns)
    cols.remove('target')
    IV_desc = sorted(IV_Var.items(),key = lambda x:x[1])
    cols = [k[0]+"_WOE" for k in IV_desc if k[0]+"_WOE" in cols]
    removedVar = []
    for k in cols:
        x0 = np.array(Data[k])
        col = [a for a in cols if (a != k and a not in removedVar)]
        X = Data[col]
        X = np.matrix(X)
        LR = LinearRegression()
        clr = LR.fit(X,x0)
        pred = clr.predict(X)
        R2 = 1 - ((pred - x0) ** 2 ).sum() / ((x0 - x0.mean()) ** 2 ).sum()
        if R2 == 1 :
            removedVar.append(k)
        else:
            vif = 1 / (1 - R2)
            print("The VIF of {} is {}".format(k,vif))
            if vif > vif_thresould:
                removedVar.append(k)
                print("     Deleted var {} for high VIF".format(k))
    cols = [k for k in cols if k not in removedVar]
    return cols


def P_Value(Data,P = 0.1):
    cols = list(Data.columns)
    cols.remove('target')
    y = Data['target']
    X = Data[cols]
    X['intercept'] = 1

    Log = sm.Logit(y,X).fit(method='bfgs')
    pvals = Log.pvalues.to_dict()

    varLargeP = { k : v for k, v in pvals.items() if v >= P }
    varLargeP = sorted(varLargeP.items(), key = lambda x:x[1],reverse=True)

    while (len(varLargeP) > 0 and len(cols) > 0 ):
        varMaxP = varLargeP[0][0]
        if varMaxP == 'intercept':
            print("The intercept is not significant!")
            break
        cols.remove(varMaxP)
        y = Data['target']
        X = Data[cols]
        X['intercept'] = 1
        Log = sm.Logit(y,X).fit()
        pvals = Log.pvalues.to_dict()
        varLargeP = {k: v for k, v in pvals.items() if v >= P}
        varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    return cols

def BestParam(Data,frac,flag = 1):
    cols = list(Data.columns)
    cols.remove('target')
    X = Data[cols]
    X = np.matrix(X)
    y = Data['target']
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=frac, random_state=0)
    print(X_train.shape)
    print(X_test.shape)
    if flag == 1:
        model_parameter = {}
        for C_penalty in np.arange(0.005, 0.2, 0.01):
            for bad_weight in range(1, 101, 5):
                LR_model_2 = LogisticRegressionCV(Cs=[C_penalty], penalty='l2', solver='liblinear',
                                                  class_weight={1: bad_weight, 0: 1})
                LR_model_2_fit = LR_model_2.fit(X_train, y_train)
                y_pred = LR_model_2_fit.predict_proba(X_train)[:, 1]
                scorecard_result = pd.DataFrame({'prob': y_pred, 'target': y_train})
                performance = KS_AR(scorecard_result, 'prob', 'target')
                KS = performance['KS']
                model_parameter[(C_penalty, bad_weight)] = KS
                print('the C_penalty is {0}, the bad_weight is {1}, the KS is {2}'.format(C_penalty, bad_weight, KS))
        Best_C, Best_badweight = max(model_parameter,key = model_parameter.get)
        print("      Best Parameters C is {0}, bad_weight is {1}, KS is {2}".format(Best_C, Best_badweight,max(model_parameter.values())))
    else:
        Best_C = 0.10
        Best_badweight = 10
    LR_model_2 = LogisticRegressionCV(Cs=[Best_C], penalty='l2', solver='liblinear',
                                      class_weight={1: Best_badweight, 0: 1})
    LR_model_2_fit = LR_model_2.fit(X_train, y_train)
    y_train_pred = LR_model_2_fit.predict_proba(X_train)[:, 1]
    result_train = pd.DataFrame({'prob':y_train_pred,'target':y_train})
    performance_train = KS_AR(result_train,'prob','target')
    auc_train = roc_auc_score(y_train,y_train_pred)
    y_test_pred = LR_model_2_fit.predict_proba(X_test)[:, 1]
    result_test = pd.DataFrame({'prob': y_test_pred, 'target': y_test})
    performance_test = KS_AR(result_test,'prob','target')
    auc_test = roc_auc_score(y_test,y_test_pred)
    print("          KS        AUC")
    print("train    %.4f   %.4f" % (performance_train['KS'],auc_train))
    print("test     %.4f   %.4f" % (performance_test['KS'], auc_test))
    return LR_model_2

def moedel_ks(data,score):
    good0 = data[data['target'] == 0].shape[0]
    bad0 = data[data['target'] == 1].shape[0]
    good1 = []
    bad1 = []
    for j in np.arange(0,1,0.001):
        good1.append(
            data[(data[score] <= j) & (data['target'] == 0)].shape[0] / good0)
        bad1.append(
            data[(data[score] <= j) & (data['target'] == 1)].shape[0] / bad0)
    good_bad = abs(np.array(good1) - np.array(bad1))
    quan_index = list(np.arange(0,1,0.001))
    gb_max = max(good_bad)
    gb_max_index = np.where(good_bad == gb_max)[0][0]
    gb_max_quan = quan_index[gb_max_index]
    print('KS:',str(round(gb_max, 3)))
    print('score',str(round(quan_index[gb_max_index], 3)))
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.plot(quan_index, good1, label='good')
    # plt.plot(quan_index, bad1, label='bad')
    # plt.plot(quan_index, good_bad, label='ks')
    # plt.plot([gb_max_quan, gb_max_quan], [0, gb_max], c='grey')
    # plt.scatter(gb_max_quan, gb_max, c='grey')
    # plt.text(gb_max_quan * 1.1, gb_max * 1.3, 'KS:' + str(round(gb_max, 3))+' where score:'+ str(round(quan_index[gb_max_index], 3)))
    # plt.title('ROC_train')
    # plt.ylabel('percentage')
    # plt.xlabel('score')
    # plt.show()

