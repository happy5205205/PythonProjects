# !/usr/bin python3.5
# -*- coding: utf-8 -*-

import sys
import datetime
import subprocess
import numpy as np
import pandas as pd
from numpy import array
import statsmodels.api as sm
from sklearn.externals import joblib
from pyspark import SparkConf, SQLContext, SparkContext, StorageLevel
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import StringIndexer, StringIndexerModel, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.linalg import Vectors
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")


# 计时函数
def tickClock(startTime):
    endTime = datetime.datetime.timestamp(datetime.datetime.now())
    costTime = round((endTime - startTime) / 60, 2)
    endTimeStr = datetime.datetime.strftime(datetime.datetime.fromtimestamp(endTime), '%Y-%m-%d %H:%M:%S')
    return endTime, endTimeStr, costTime


# 合并分裂的数据集
def splitedDataMerge(fn, fnList, dataSetType):
    dataSetWoe = ''
    featureIvMap = {}
    keepFeaturesTmp = []
    for i, fni in enumerate(fnList):
        if i == 0:
            dataSetWoe = sc.textFile(savePath + '{}/{}/{}/{}SetWoe'.format(curDate, fn, fni, dataSetType)).map(
                lambda a: (a[:a.find(',')], a[(a.find(',') + 1):])).repartition(
                numExecutors).persist(storageLevel)
            featureIvMap = joblib.load(localPath + '{}/{}/{}/featureIvMap.pkl'.format(curDate, fn, fni))
            keepFeaturesTmp = joblib.load(localPath + '{}/{}/{}/keepFeaturesFinal.pkl'.format(curDate, fn, fni))
        else:
            dataSetWoeTmp = sc.textFile(savePath + '{}/{}/{}/{}SetWoe'.format(curDate, fn, fni, dataSetType)).map(
                lambda a: (a[:a.find(',')], a[(a.find(',') + 1):])).repartition(
                numExecutors).persist(storageLevel)
            dataSetWoe = dataSetWoe.join(dataSetWoeTmp).map(lambda x: (x[0], ','.join(x[1])))
            featureIvMap.update(joblib.load(localPath + '{}/{}/{}/featureIvMap.pkl'.format(curDate, fn, fni)))
            keepFeaturesTmp = keepFeaturesTmp + joblib.load(
                localPath + '{}/{}/{}/keepFeaturesFinal.pkl'.format(curDate, fn, fni))
    return dataSetWoe, featureIvMap, keepFeaturesTmp


# 将数据集筛选入模型变量并转为rdd.df
def loadData2DF(dataSetWoe, keepFeaturesByCorr, keepFeaturesFinal):
    dataSetWoeDF = dataSetWoe.mapPartitions(
        lambda f: map(lambda x: [x[0]] + [float(x[0].split('_')[3])] + [float(i) for i in x[1].split(',')], f)). \
        toDF(['id_key'] + ['target'] + keepFeaturesByCorr).select(['id_key'] + ['target'] + keepFeaturesFinal)
    return dataSetWoeDF


# 模型训练
def modelTraining(trainSetWoeDF, weightBalance, fn):
    # 数据预转换,满足ML-linearRegression输入格式要求
    trainSetVecAse = vecAseembler.transform(trainSetWoeDF)
    strInd = stringIndexer.fit(trainSetVecAse)
    trainSetVecAseStrInd = strInd.transform(trainSetVecAse)
    trainSetVecAseStrIndWet = trainSetVecAseStrInd.withColumn('weight', trainSetVecAseStrInd.target * weightBalance + 1)
    # 模型训练
    lrm = LogisticRegression(regParam=0.01, weightCol="weight")
    lrModel = lrm.fit(trainSetVecAseStrIndWet)
    trainSetWithProba = lrModel.transform(trainSetVecAseStrIndWet)
    # 保存模型及相关参数
    vecAseembler.write().overwrite().save(savePath + '{}/{}/vecAseembler'.format(curDate, fn))
    strInd.write().overwrite().save(savePath + '{}/{}/strInd'.format(curDate, fn))
    lrModel.write().overwrite().save(savePath + '{}/{}/lrModel'.format(curDate, fn))
    # joblib.dump([lr_model.intercept, lr_model.coefficients], localPath + 'params/lrFinalCoef_{}.pkl'.format(fn))
    coefNotNegtive = np.where(lrModel.coefficients.toArray() > 0)[0]
    return (trainSetWithProba, coefNotNegtive)


# 模型预测
def modelPredicting(testSetWoeDF, fn):
    # 数据预转换,满足ML-linearRegression输入格式要求
    strInd = StringIndexerModel.load(savePath + '{}/{}/strInd'.format(curDate, fn))
    lrModel = LogisticRegressionModel.load(savePath + '{}/{}/lrModel'.format(curDate, fn))
    testSetVecAse = vecAseembler.transform(testSetWoeDF)
    testSetVecAseStrInd = strInd.transform(testSetVecAse)
    testSetWithProba = lrModel.transform(testSetVecAseStrInd)
    return (testSetWithProba)


# 重分组方法
def AssignGroup(x, bin):
    # 后续考虑添加缺失值处理
    N = len(bin)
    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return np.inf
    else:
        for i in range(N - 1):
            if bin[i] < x <= bin[i + 1]:
                return bin[i + 1]


# rdd.df转为hive表
def dfSave2Hive(dataSetWithProba, keepFeaturesFinal, dataType):
    dataSetWithProbaDF = dataSetWithProba.select(['id_key', 'probability', 'features']).rdd.map(
        lambda x: [x[0], str(x[1][0])] + [str(i) for i in array(x[2])]). \
        toDF(['id_key', 'probability'] + keepFeaturesFinal)
    dataSetWithProbaDF.write.mode('overwrite').saveAsTable(
        "xy_linsanji.{}_{}_{}".format(fn, curDate.replace('-', ''), dataType))


# 计算KS值
def modelEva(trainSetWithProba):
    cutoffpoint = list(np.arange(0, 1, .05))  # [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    trainSetProbaCutBins = trainSetWithProba.select(['probability', 'target']).rdd. \
        mapPartitions(
        lambda f: map(lambda x: (AssignGroup(x[0][1], cutoffpoint),
                                 ((float(x[1]) == 0) * 1, (float(x[1]) == 1) * 1)), f)). \
        reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    trainSetKsMat = trainSetProbaCutBins. \
        sortByKey(). \
        map(lambda x: (float(x[0]), float(x[1][0]), float(x[1][1]))). \
        toDF(). \
        toPandas()
    trainSetKsMat['goodRate'] = trainSetKsMat['_2'].cumsum() / trainSetKsMat['_2'].sum()
    trainSetKsMat['badRate'] = trainSetKsMat['_3'].cumsum() / trainSetKsMat['_3'].sum()
    trainSetKsMat['ks'] = abs(trainSetKsMat['goodRate'] - trainSetKsMat['badRate'])
    trainSetKsMat['totalBinRate'] = (trainSetKsMat['_2'] + trainSetKsMat['_3']) / (
            trainSetKsMat['_2'].sum() + trainSetKsMat['_3'].sum())
    trainSetKsMat['passRate'] = trainSetKsMat['totalBinRate'].cumsum()
    trainSetKsMat['badPassRate'] = trainSetKsMat['_3'].cumsum() / (
            trainSetKsMat['_2'].cumsum() + trainSetKsMat['_3'].cumsum())
    colDict = {'_1': 'proba', '_2': '#goods', '_3': '#bads'}
    trainSetKsMat = trainSetKsMat.rename(index=str, columns=colDict)
    # 其他指标测试
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="target")
    areaUnderPR = evaluator.evaluate(trainSetWithProba, {evaluator.metricName: "areaUnderPR"})
    areaUnderROC = evaluator.evaluate(trainSetWithProba, {evaluator.metricName: "areaUnderROC"})
    metricsResult = {
        'roc': round(areaUnderROC, 3),
        'pr': round(areaUnderPR, 3),
        'ks': round(trainSetKsMat['ks'].max(), 3),
        'tpr': round(trainSetKsMat[trainSetKsMat['ks'] == trainSetKsMat['ks'].max()]['goodRate'].values[0], 3),
        'ksmatrix': trainSetKsMat
    }
    # joblib.dump(metricsResult, localPath + '{}/{}/metricsResult.pkl'.format(curDate, fn))
    return (metricsResult)

def modelEva2(trainSetWithProba):
    cutoffpoint = list(np.arange(0, 1, .05))  # [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    trainSetProbaCutBins = trainSetWithProba.select(['probability', 'target']).rdd. \
        mapPartitions(
        lambda f: map(lambda x: (AssignGroup(x[0][1], cutoffpoint),
                                 ((float(x[1]) == 0) * 1, (float(x[1]) != 0) * 1)), f)). \
        reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    trainSetKsMat = trainSetProbaCutBins. \
        sortByKey(). \
        map(lambda x: (float(x[0]), float(x[1][0]), float(x[1][1]))). \
        toDF(). \
        toPandas()
    trainSetKsMat['goodRate'] = trainSetKsMat['_2'].cumsum() / trainSetKsMat['_2'].sum()
    trainSetKsMat['badRate'] = trainSetKsMat['_3'].cumsum() / trainSetKsMat['_3'].sum()
    trainSetKsMat['ks'] = abs(trainSetKsMat['goodRate'] - trainSetKsMat['badRate'])
    trainSetKsMat['totalBinRate'] = (trainSetKsMat['_2'] + trainSetKsMat['_3']) / (
            trainSetKsMat['_2'].sum() + trainSetKsMat['_3'].sum())
    trainSetKsMat['passRate'] = trainSetKsMat['totalBinRate'].cumsum()
    trainSetKsMat['badPassRate'] = trainSetKsMat['_3'].cumsum() / (
            trainSetKsMat['_2'].cumsum() + trainSetKsMat['_3'].cumsum())
    colDict = {'_1': 'proba', '_2': '#goods', '_3': '#bads'}
    trainSetKsMat = trainSetKsMat.rename(index=str, columns=colDict)
    # 其他指标测试
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="target")
    areaUnderPR = evaluator.evaluate(trainSetWithProba, {evaluator.metricName: "areaUnderPR"})
    areaUnderROC = evaluator.evaluate(trainSetWithProba, {evaluator.metricName: "areaUnderROC"})
    metricsResult = {
        'roc': round(areaUnderROC, 3),
        'pr': round(areaUnderPR, 3),
        'ks': round(trainSetKsMat['ks'].max(), 3),
        'tpr': round(trainSetKsMat[trainSetKsMat['ks'] == trainSetKsMat['ks'].max()]['goodRate'].values[0], 3),
        'ksmatrix': trainSetKsMat
    }
    # joblib.dump(metricsResult, localPath + '{}/{}/metricsResult.pkl'.format(curDate, fn))
    return (metricsResult)


# 计算psi
def psi(psi_train, psi_test):
    c = pd.concat([psi_train.set_index('proba'), psi_test.set_index('proba')], axis=1).fillna(0.00001)
    d = c / c.sum(axis=0)
    d.columns = ['ref', 'val']
    return (((d.val - d.ref) * np.log(d.val / d.ref)).sum())


if __name__ == "__main__":
    conf = SparkConf(). \
        set("spark.shuffle.consolidateFiles", "true"). \
        set("spark.serializer", "org.apache.spark.serializer.KryoSerializer"). \
        set("spark.default.parallelism", "5")
    sc = SparkContext(appName='SCP2_{}'.format(sys.argv[1]), conf=conf)
    sqlContext = SQLContext(sc)
    savePath = '/user/hive/warehouse/xy_linsanji.db/scoreCardRegress/'
    localPath = '/home/xy_jiangyuande/lsj/scoreCardRegress/'

    # curDate = '2018-06-04'
    # fnList = 'externalPayXY_oxy4into1_0601_merge_test,externalBorrXY_oxy4into1_0601_merge_test'.split(',')
    # rohThreshold = 0.6
    # vifThreshold = 5.0
    # weightBalance = 1
    # numExecutors = 150
    # fn = 'oxy4into1_0601_merge_test'

    fn = sys.argv[1]
    curDate = sys.argv[2]
    if curDate == '1':
        curDate = str(datetime.datetime.now()).split(' ')[0]
    # fnList = sys.argv[1].split('#')
    fnList = ['borrbasic1', 'borrbasic2', 'borrex', 'payex', 'paybasic']
    numExecutors = int(sys.argv[3]) * len(fnList)  # 合并资源
    rohThreshold = float(sys.argv[4])
    vifThreshold = float(sys.argv[5])
    weightBalance = int(sys.argv[6])
    storageLevel = StorageLevel(True, True, False, False, 1)
    startTime = datetime.datetime.timestamp(datetime.datetime.now())
    startTimeStr = datetime.datetime.strftime(datetime.datetime.fromtimestamp(startTime), '%Y-%m-%d %H:%M:%S')

    # subprocess.call(["hadoop", "fs", "-rm", "-r", savePath + '{}/{}'.format(curDate, fn)], stdin=False, stdout=False)
    # subprocess.call(["mkdir", localPath + '{}/{}'.format(curDate, fn)], stdin=False, stdout=False)

    print('==================================== Welcome to Lin\'s world ^-^ ! ====================================')
    print('{} #5.  Start merge splited data ...'.format(startTimeStr))
    trainWoe, featureIvMap, keepFeaturesByCorr = splitedDataMerge(fn, fnList, 'train')
    devSetWoe, _, _ = splitedDataMerge(fn, fnList, 'dev')
    ootSetWoe, _, _ = splitedDataMerge(fn, fnList, 'oot')
    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #5.  Finish merge splited data ! Total merged #{} feats ! Cost {} mins !'.format(endTimeStr,
                                                                                               len(keepFeaturesByCorr),
                                                                                               costTime))
    # 剔除高相关性低iv变量
    print('{} #6.  Start Del higher than {} Corr feats...'.format(endTimeStr, rohThreshold))
    trainWoeCo = trainWoe.map(lambda x: [float(i) for i in x[1].split(',')]).persist(storageLevel)
    removedVar = []
    loop_circle = len(keepFeaturesByCorr)
    pearsonCorr = Statistics.corr(trainWoeCo, method='pearson')
    for i in range(loop_circle - 1):
        for j in range(i + 1, loop_circle):
            if j > i:
                if abs(pearsonCorr[i, j]) >= rohThreshold \
                        and keepFeaturesByCorr[i] not in removedVar \
                        and keepFeaturesByCorr[j] not in removedVar:
                    # 选择IV高的变量保留,剔除另一个变量
                    if featureIvMap[keepFeaturesByCorr[i]] > featureIvMap[keepFeaturesByCorr[j]]:
                        removedVar.append(keepFeaturesByCorr[j])
                    else:
                        removedVar.append(keepFeaturesByCorr[i])
    # 保留变量
    keepFeaturesFinal = [i for i in keepFeaturesByCorr if i not in removedVar]  # 是否会打乱顺序？
    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #6.  Finish Del high Corr feats ! Keep #{} ! Cost {} mins !'.format(endTimeStr, len(keepFeaturesFinal),
                                                                                  costTime))

    # 剔除高共线性VIF的指标
    dataSetWoeDF = trainWoe.mapPartitions(
        lambda f: map(lambda x: [float(x[0].split('_')[3])] + [float(i) for i in x[1].split(',')], f)). \
        toDF(['target'] + keepFeaturesByCorr).select(['target'] + keepFeaturesFinal).toPandas()
    regr = LinearRegression()
    print('{} #7.  Start Del Vif higher than {} feats...'.format(endTimeStr, vifThreshold))
    for var in keepFeaturesFinal:
        x0 = np.array(dataSetWoeDF[var])
        restVar = [k for k in keepFeaturesFinal if k != var]
        X = np.matrix(dataSetWoeDF[restVar])
        clr = regr.fit(X, x0)
        x_pred = clr.predict(X)
        R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
        vif = 1 / (1 - R2)
        if vif > vifThreshold: removedVar.append(var)
    keepFeaturesFinal = [i for i in keepFeaturesFinal if i not in removedVar]
    startTime, endTimeStr, costTime = tickClock(startTime)
    print(
        '{} #7.  Finish Del Insignificant feats ! Keep #{} ! Cost {} mins !'.format(endTimeStr, len(keepFeaturesFinal),
                                                                                    costTime))
    ### 借助逻辑回归检验变量显著性
    print('{} #8.  Start Del PValue higher than 0.1 feats...'.format(endTimeStr))
    y = dataSetWoeDF['target']
    X = dataSetWoeDF[keepFeaturesFinal]
    X['intercept'] = [1] * X.shape[0]
    LR = sm.Logit(y, X).fit(disp=False)
    pvals = LR.pvalues
    pvals = pvals.to_dict()

    ### 删除不显著变量
    varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
    varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    while (len(varLargeP) > 0 and len(keepFeaturesFinal) > 0):
        # 每次迭代中，剔除最不显著的变量，直到
        # (1) 剩余所有变量均显著
        # (2) 没有特征可选
        varMaxP = varLargeP[0][0]
        if varMaxP == 'intercept':
            print('the intercept is not significant!')
            break
        keepFeaturesFinal.remove(varMaxP)
        y = dataSetWoeDF['target']
        X = dataSetWoeDF[keepFeaturesFinal]
        X['intercept'] = [1] * X.shape[0]
        LR = sm.Logit(y, X).fit(disp=False)
        pvals = LR.pvalues
        pvals = pvals.to_dict()
        varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
        varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    startTime, endTimeStr, costTime = tickClock(startTime)
    joblib.dump(keepFeaturesFinal, localPath + '{}/{}/keepFeaturesFinal.pkl'.format(curDate, fn))
    print(
        '{} #8.  Finish Del Insignificant feats ! Keep #{} ! Cost {} mins !'.format(endTimeStr, len(keepFeaturesFinal),
                                                                                    costTime))
    # # 剔除高共线性VIF的指标
    # lrm = LinearRegression(maxIter=10, regParam=0.1, solver="auto")
    # print('{} #7.  Start Del higher than {} Vif feats...'.format(endTimeStr, vifThreshold))
    # for i, var in enumerate(keepFeaturesByCorr):
    #     if var in keepFeaturesFinal:
    #         curIdx = [idx for idx, col in enumerate(keepFeaturesByCorr) if col not in removedVar and col not in [var]]
    #         trainWoeDF = trainWoeCo.map(lambda x: (x[i], Vectors.dense(np.array(x)[curIdx]))).toDF(
    #             ['label', 'features'])
    #         mdP = lrm.fit(trainWoeDF).transform(trainWoeDF)
    #         xMean = mdP.groupby().mean('label').collect()[0][0]
    #         SSE = mdP.rdd.map(lambda x: (x['prediction'] - x['label']) ** 2).sum()
    #         SST = mdP.rdd.map(lambda x: (x['label'] - xMean) ** 2).sum()
    #         R2 = 1 - SSE / SST
    #         vif = 1 / (1 - R2)
    #         if vif > vifThreshold: removedVar.append(var)
    # 保留变量
    # keepFeaturesFinal = [i for i in keepFeaturesByCorr if i not in removedVar]
    # joblib.dump(keepFeaturesFinal, localPath + '{}/{}/keepFeaturesFinal.pkl'.format(curDate, fn))
    # startTime, endTimeStr, costTime = tickClock(startTime)
    # print('{} #8.  Finish Del high Vif feats ! Keep #{} ! Cost {} mins !'.format(endTimeStr, len(keepFeaturesFinal),
    #                                                                              costTime))
    # 剔除非显著性P<0.05的指标
    # from pyspark.mllib.stat import Statistics
    # trainSetWoeP = trainSetWoe.map(lambda x: LabeledPoint(int(x[0].split('_')[3]),array([float(i) for i in x[1].split(',')]))).persist(storageLevel)
    # chi = Statistics.chiSqTest(trainSetWoeP)
    # import numpy as np
    # import statsmodels.api as sm
    # data = sm.datasets.longley.load()
    # data.exog = sm.add_constant(data.exog)
    # results = sm.OLS(data.endog, data.exog).fit()
    # r = np.eye(len(results.params))
    # T_test = results.t_test(r)
    # T_test.pvalue>0.05

    print('{} #9.  Start model build ...'.format(endTimeStr))

    trainSetWoeDF = loadData2DF(trainWoe, keepFeaturesByCorr, keepFeaturesFinal)
    devSetWoeDF = loadData2DF(devSetWoe, keepFeaturesByCorr, keepFeaturesFinal)
    ootSetWoeDF = loadData2DF(ootSetWoe, keepFeaturesByCorr, keepFeaturesFinal)

    # 设置模型参数,建模
    vecAseembler = VectorAssembler(inputCols=keepFeaturesFinal, outputCol='features')
    stringIndexer = StringIndexer(inputCol='target', outputCol='label')
    trainSetWithProba, coefNotNegtive = modelTraining(trainSetWoeDF, weightBalance, fn)
    devSetWithProba = modelPredicting(devSetWoeDF, fn)
    ootSetWithProba = modelPredicting(ootSetWoeDF, fn)

    dfSave2Hive(trainSetWithProba, keepFeaturesFinal, 'train')
    dfSave2Hive(devSetWithProba, keepFeaturesFinal, 'dev')
    dfSave2Hive(ootSetWithProba, keepFeaturesFinal, 'oot')

    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #9.  Finish model build ! Total have #{} not negtive coefs ! Cost {} mins.'.format(endTimeStr,
                                                                                                 len(coefNotNegtive),
                                                                                                 costTime))
    # 保留预测及真实结果，计算ROC\PR\KS\PSI
    print('{} #10.  Start model evaluation...'.format(endTimeStr))
    trainSetEva = modelEva(trainSetWithProba)
    devSetEva = modelEva(devSetWithProba)
    ootSetEva = modelEva(ootSetWithProba)
    trainSetEva['psi'] = 0
    devSetEva['psi'] = psi(trainSetEva['ksmatrix'][['proba', 'totalBinRate']],
                           devSetEva['ksmatrix'][['proba', 'totalBinRate']])
    ootSetEva['psi'] = psi(trainSetEva['ksmatrix'][['proba', 'totalBinRate']],
                           ootSetEva['ksmatrix'][['proba', 'totalBinRate']])

    trainMat = trainSetEva['ksmatrix'][['proba', 'badPassRate', 'passRate']].set_index('proba')
    devMat = devSetEva['ksmatrix'][['proba', 'badPassRate', 'passRate']].set_index('proba')
    ootMat = ootSetEva['ksmatrix'][['proba', 'badPassRate', 'passRate']].set_index('proba')
    d = pd.concat([trainMat, devMat, ootMat], axis=1)
    d.columns = ['trainBad%', 'trainThrogh%', 'devBad%', 'devThrogh%', 'ootBad%', 'ootThrogh%']
    del trainSetEva['ksmatrix']
    del devSetEva['ksmatrix']
    del ootSetEva['ksmatrix']
    a = pd.DataFrame(pd.Series(trainSetEva), columns=['trainSetEva'])
    b = pd.DataFrame(pd.Series(devSetEva), columns=['devSetEva'])
    c = pd.DataFrame(pd.Series(ootSetEva), columns=['ootSetEva'])
    print(pd.concat([a, b, c], axis=1).T)
    print(d)
    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #10.  Finish model evaluation ! Cost {} mins.'.format(endTimeStr, costTime))
    print('==================================== Have Fun ! See ya next time ^-^ ! ====================================')

