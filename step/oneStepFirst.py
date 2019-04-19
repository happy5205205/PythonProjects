# !/usr/bin python3.5
# -*- coding: utf-8 -*-

import sys
import random
import datetime
import threading
import subprocess
import numpy as np
import pandas as pd
from operator import add
from multiprocessing import Queue
from sklearn.externals import joblib
from pyspark import SparkConf, SQLContext, SparkContext, StorageLevel
from pyspark.mllib.stat import Statistics
import warnings

warnings.filterwarnings("ignore")


# 分裂全变量下标，根据并行度做拆分
def SplitFeatures(all_features, split_size):
    if (split_size == 1) or len(all_features) <= split_size:
        return [all_features]
    elif split_size == 2:
        middle = int(len(all_features) / 2)
        return [all_features[:middle]] + [all_features[middle:]]
    else:
        feature_length = len(all_features)
        step_interval = int(feature_length / split_size)
        start_index = 0
        end_index = step_interval
        arr = [all_features[start_index:end_index]]
        for i in range(1, split_size):
            start_index = end_index
            end_index = start_index + step_interval
            arr.append(all_features[start_index:end_index])
        start_index = end_index
        if start_index < feature_length:
            interval = feature_length - start_index
            arr.append(all_features[start_index:start_index + interval])
        return arr


# 计算特征IV方法
def IviFunc(a, b, goodTotal, badTotal):
    return (((1.0 if a == 0 else a * 1.0) / goodTotal) - ((1.0 if b == 0 else b * 1.0) / badTotal)) * np.log(
        ((1.0 if a == 0 else a * 1.0) / goodTotal) * 1.0 / ((1.0 if b == 0 else b * 1.0) / badTotal))


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


# 计算卡方值
def Chi2Map(xMap, overallRate, groupIntervals):
    grpIndex = [x for x, y in zip(range(len(groupIntervals)), groupIntervals) if xMap[0] in y][0]
    chi2 = (sum(xMap[1]) * overallRate - xMap[1][1]) ** 2 / (sum(xMap[1]) * overallRate)
    return (grpIndex, chi2)


# 卡方分箱方法
def Chi2Merge(chisqList, intervalReGroup, groupNum):
    # 对最小卡方值的区间,找出卡方值最接近的的区间,并合并相邻区间
    min_position = chisqList.index(min(chisqList))
    if min_position == 0:
        # 若最小卡方值所在区间为1,则合并后一个区间-:2
        combinedPosition = 1
    elif min_position == groupNum - 1:
        # 若最小卡方值所在区间为倒二,则合并最后一个区间-:-1
        combinedPosition = min_position - 1
    else:
        if chisqList[min_position - 1] <= chisqList[min_position + 1]:
            # 若最小卡方值区间的,上一个区间卡方值,小于等于,后一个区间卡方值,则合并上一个区间
            combinedPosition = min_position - 1
        else:
            # 否则合并后一个区间
            combinedPosition = min_position + 1
    # 合并的卡方最小的周边区间
    intervalReGroup[min_position] = sorted(intervalReGroup[min_position] + intervalReGroup[combinedPosition])
    chisqList[min_position] = chisqList[min_position] + chisqList[combinedPosition]
    # 删除被合并的区间
    intervalReGroup.remove(intervalReGroup[combinedPosition])
    chisqList.remove(chisqList[combinedPosition])
    return chisqList, intervalReGroup


# 几个转折点(单调变量没有转折点,允许二次型变量)
def monotonecheck(vlist):
    valnum = 0
    valraw = vlist[0]
    for i in vlist:
        if valraw != i:
            valraw = i
            valnum += 1
    return valnum


# 单调性检验方法
def MonotoneCheck(dfReGroup, intervalReGroup):
    cutOffPoints = [i[-1] for i in intervalReGroup[:-1]]  # 对合并后的区间,划分点为合并区间最后一个取值
    badRate = dfReGroup. \
        mapPartitions(lambda f: map(lambda x: (AssignGroup(x[0], cutOffPoints), x[1]), f)). \
        reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])). \
        sortByKey(). \
        map(lambda x: x[1][1] / sum(x[1])). \
        collect()
    # 计算单调性
    badRateMonotone = [badRate[i] < badRate[i + 1] for i in range(len(badRate) - 1)]
    # badRateMonotone = [False, False, True, True, True, False]
    Monotone = monotonecheck(badRateMonotone) <= 1
    # 几个转折点(单调变量没有转折点,只允许一次型变量)
    # Monotone = badRateMonotone.count(True) == 1 or badRateMonotone.count(False) == 1
    return Monotone


# 计算特征WOE方法
def WoeFunc(a, b, goodTotal, badTotal, ndigit=4):
    return round(float(np.log(
        ((1.0 if a == 0 else a * 1.0) / goodTotal) * 1.0 /
        ((1.0 if b == 0 else b * 1.0) / badTotal))), ndigit)


# 重分组方法
def AssignGroupWOE(x, cutOffBin, woeMap):
    # 后续考虑添加缺失值处理
    N = len(cutOffBin)
    if x <= min(cutOffBin):
        return woeMap[0]
    elif x > max(cutOffBin):
        return woeMap[-1]
    else:
        for i in range(N - 1):
            if cutOffBin[i] < x <= cutOffBin[i + 1]:
                return woeMap[i + 1]


# 分箱占比校验方法
def MaxIntervalPercentCheck(dfReGroup, groupNum, intervalReGroup, IV_before, Total, goodTotal, badTotal):
    if groupNum == 1:
        cutOffPoints = [i[-1] for i in intervalReGroup]
    else:
        cutOffPoints = [i[-1] for i in intervalReGroup[:-1]]
    dfReAssignGroup = dfReGroup. \
        mapPartitions(lambda f: map(lambda x: (AssignGroup(x[0], cutOffPoints), x[1]), f)). \
        reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])). \
        cache()
    pMaxCnt = dfReAssignGroup.map(lambda x: sum(x[1])).max()
    if pMaxCnt / Total > 0.9:
        return (([], IV_before, 0., []))
    else:
        IV_current = dfReAssignGroup. \
            mapPartitions(
            lambda f: map(lambda x: IviFunc(x[1][0], x[1][1], goodTotal, badTotal) if all(x[1]) \
                else IviFunc(x[1][0] + 1, x[1][1] + 1, goodTotal, badTotal), f)). \
            sum()
        WoeMap = dfReAssignGroup. \
            mapPartitions(lambda f: map(lambda x: (x[0], WoeFunc(x[1][0], x[1][1], goodTotal, badTotal)), f)). \
            sortByKey().map(lambda x: x[1]).collect()
    return ((cutOffPoints, IV_before, IV_current, WoeMap))


# 非零区间抽取
def IntervalNotZero(dfGroupByKey, maxElements):
    # 剔除分组中存在计数为0的组，重新分组
    intervalIndexNotEqualZero = sorted(dfGroupByKey.filter(lambda x: all(x[1])).keys().collect())
    # 如果是常量，则跳出IV计算
    if len(intervalIndexNotEqualZero) <= 1:
        return ([])
    elif len(intervalIndexNotEqualZero) > maxElements:
        ind_x = [int(i / maxElements * len(intervalIndexNotEqualZero)) for i in range(1, maxElements)]
        intervalIndexNotEqualZero = [intervalIndexNotEqualZero[i] for i in ind_x]
    else:
        intervalIndexNotEqualZero = intervalIndexNotEqualZero
    return intervalIndexNotEqualZero


# 定义卡方分箱函数
def FeatureRoughSelect(dfCache, curr_, goodTotal, badTotal, Total, overallRate, maxInterval, maxElements,
                       minIvThreshold):  # iv=0.1 or 0.15 when in Fsloan, Reloan is 0.3
    # 计算无分组的好坏用户数
    dfGroupByKey = dfCache. \
        mapPartitions(lambda f: map(lambda x: (x[curr_], ((x[0] == 0) * 1, (x[0] == 1) * 1)), f)). \
        reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])). \
        cache()
    # 剔除分组中存在计数为0的组，重新分组
    intervalIndexNotEqualZero = IntervalNotZero(dfGroupByKey, 100)  # 默认初始100区间
    if len(intervalIndexNotEqualZero) == 0:
        return ((intervalIndexNotEqualZero, 0., 0., []))
    # 计算重新分组后的好坏用户数
    dfReGroup = dfGroupByKey. \
        mapPartitions(
        lambda f: map(lambda x: (AssignGroup(x[0], intervalIndexNotEqualZero), x[1]), f)). \
        reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])). \
        cache()
    # IV最大区间处若存在0值，合并存在问题（全加1），待优化 ---- 随机精分箱改为百分位数分箱
    IV_before = dfReGroup. \
        mapPartitions(lambda f: map(lambda x: IviFunc(x[1][0], x[1][1], goodTotal, badTotal) if all(x[1]) \
        else IviFunc(x[1][0] + 1, x[1][1] + 1, goodTotal, badTotal), f)). \
        reduce(add)
    # IV不满足基础iv筛选条件则跳出精分箱，否则继续进行粗分箱
    if IV_before < minIvThreshold:
        return (([], 0., 0., []))
    # 针对满足条件的重新做粗分箱 ---- 卡方分箱
    intervalIndexNotEqualZero = IntervalNotZero(dfGroupByKey, maxElements)  # 默认初始100区间
    dfReGroup = dfGroupByKey. \
        mapPartitions(
        lambda f: map(lambda x: (AssignGroup(x[0], intervalIndexNotEqualZero), x[1]), f)). \
        reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])). \
        cache()
    # 计算变量单个属性对应单区间的值,做出合并
    intervalReGroup = [[i] for i in sorted(dfReGroup.keys().collect())]
    groupNum = len(intervalReGroup)
    # 1.基于初始设定的最大分箱值，采用卡方统计量进行循环合并区间
    chisqlist = dfReGroup. \
        mapPartitions(lambda f: map(lambda x: Chi2Map(x, overallRate, intervalReGroup), f)). \
        reduceByKey(add). \
        sortByKey(). \
        map(lambda x: float(x[1])). \
        collect()
    while (groupNum > maxInterval):
        chisqlist, intervalReGroup = Chi2Merge(chisqlist, intervalReGroup, groupNum)
        groupNum = len(intervalReGroup)
    ## 2.基于单调性定义，针对不单调的(MC == False)逐次合并分箱区间
    MC = MonotoneCheck(dfReGroup, intervalReGroup)
    if MC:
        return MaxIntervalPercentCheck(dfReGroup, groupNum, intervalReGroup, IV_before, Total, goodTotal, badTotal)
    while not MC:
        chisqList, intervalReGroup = Chi2Merge(chisqlist, intervalReGroup, groupNum)
        groupNum = len(intervalReGroup)
        if groupNum == 1:
            MC = True
        else:
            MC = MonotoneCheck(dfReGroup, intervalReGroup)
        ## 3.如果单调(MC == True)，继续判断单个分箱样本是否占比过大，过大则跳出，废弃该变量
        if MC:
            return MaxIntervalPercentCheck(dfReGroup, groupNum, intervalReGroup, IV_before, Total, goodTotal, badTotal)


# 变量粗筛队列
def FeatureRoughSelectQueue(dataTrain, paramsDict):
    currFeatures, sep, maxInterval, maxElements, minIvThreshold = paramsDict
    # 持久化当前特征的数据集
    dfCorrTmp = dataTrain. \
        mapPartitions(lambda f: map(lambda x:
                                    [round(float(i), 4) for i in x.split(sep)[currFeatures[0]:(currFeatures[-1] + 1)]],
                                    f))
    # 单文件相关性（硬）处理，随机剔除高相关性变量
    removedVar = []
    loop_circle = len(currFeatures)
    pearsonCorr = Statistics.corr(dfCorrTmp, method='pearson')
    for i in range(loop_circle - 1):
        for j in range(i + 1, loop_circle):
            if j > i:
                if abs(pearsonCorr[i, j]) >= rohThreshold and currFeatures[i] not in removedVar:
                    removedVar.append(currFeatures[i])
    # 持久化当前特征的数据集
    dfCache = dataTrain. \
        mapPartitions(lambda f: map(lambda x:
                                    [int(float(x.split(sep)[0].split('_')[3]))] +
                                    [round(float(i), 4) for i in x.split(sep)[currFeatures[0]:(currFeatures[-1] + 1)]],
                                    f)).persist(storageLevel)
    # 循环遍历该并行块的所有变量，做变量粗筛
    res = []
    dfCache.take(1)
    for curr_, currName in zip(range(1, len(currFeatures) + 1), currFeatures):
        if currName not in removedVar:
            res.append(
                [currName,
                 FeatureRoughSelect(dfCache, curr_, goodTotal, badTotal, Total, overallRate, maxInterval, maxElements,
                                    minIvThreshold)])
        else:
            res.append([currName, ([], 0., 0., [])])
    dfCache.unpersist()
    taskQueue.put(res)


# 计时函数
def tickClock(startTime):
    endTime = datetime.datetime.timestamp(datetime.datetime.now())
    costTime = round((endTime - startTime) / 60, 2)
    endTimeStr = datetime.datetime.strftime(datetime.datetime.fromtimestamp(endTime), '%Y-%m-%d %H:%M:%S')
    return endTime, endTimeStr, costTime


# 保存woe数据
def saveWoe(path, numSaveRepartitions, keepFeaturesFinal, featureCutOffPointMap, featureWoeMap):
    dataSetWoe = sc.textFile(path).repartition(numSaveRepartitions). \
        mapPartitions(lambda f: map(lambda x: x.split(sepStr)[0] + ',' + ','.join([
        str(AssignGroupWOE(float(x.split(sepStr)[int(real_curr.split('_')[-1])]), featureCutOffPointMap[real_curr],
                           featureWoeMap[real_curr]))
        for real_curr in list(keepFeaturesFinal)]), f)).persist(storageLevel)
    return dataSetWoe


# 保存raw数据
def saveRaw(path, numSaveRepartitions, keepFeaturesFinal):
    dataSetRaw = sc.textFile(path).repartition(numSaveRepartitions). \
        mapPartitions(lambda f: map(lambda x: x.split(sepStr)[0] + ',' + ','.join([
        str(x.split(sepStr)[int(real_curr.split('_')[-1])])
        for real_curr in list(keepFeaturesFinal)]), f)).persist(storageLevel)
    return dataSetRaw


if __name__ == "__main__":
    conf = SparkConf(). \
        set("spark.shuffle.consolidateFiles", "true"). \
        set("spark.serializer", "org.apache.spark.serializer.KryoSerializer"). \
        set("spark.default.parallelism", "5")
    sc = SparkContext(appName='SCP1_{}_{}'.format(sys.argv[1], sys.argv[2]), conf=conf)
    sqlContext = SQLContext(sc)
    # 数据路径
    savePath = '/user/hive/warehouse/xy_linsanji.db/scoreCardRegress/'
    localPath = '/home/xy_jiangyuande/lsj/scoreCardRegress/'

    # fn = 'haier_0627'
    # subFn = 'payex'
    # sepStr = ','
    # featsType = 'hdfs'
    # curDate = '2018-06-27'
    # numThreads = 10
    # numExecutors = 20
    # numLoopParts = 10
    # maxInterval = 20
    # maxElements = 1000
    # minIvThreshold = 0.05
    # rohThreshold = 0.8
    # trainSetPercentage = 100
    # ootSplitDate = '2018-05-01'
    # diffStrategy = '0'

    fn = sys.argv[1]
    subFn = sys.argv[2]
    sepStr = sys.argv[3]
    featsType = sys.argv[4]
    curDate = sys.argv[5]
    if curDate == '1':
        curDate = str(datetime.datetime.now()).split(' ')[0]
    numThreads = int(sys.argv[6])
    numExecutors = int(sys.argv[7])
    numLoopParts = int(sys.argv[8])
    maxInterval = int(sys.argv[9])
    maxElements = int(sys.argv[10])
    minIvThreshold = float(sys.argv[11])
    rohThreshold = float(sys.argv[12])
    trainSetPercentage = int(sys.argv[13])
    ootSplitDate = sys.argv[14]
    diffStrategy = '1'
    if ootSplitDate == '1':
        diffStrategy = '0'
    storageLevel = StorageLevel(True, True, False, False, 1)

    # 续上#2回，注释部分#
    subprocess.call(["kinit", "-kt", "/opt/xy_keytab/xy_jiangyuande.keytab", "xy_jiangyuande"])
    subprocess.call(["hadoop", "fs", "-rm", "-r", savePath + '{}/{}/{}'.format(curDate, fn, subFn)], stdin=False,
                    stdout=False)
    subprocess.call(["rm", "-r", localPath + '{}/{}/{}'.format(curDate, fn, subFn)], stdin=False, stdout=False)
    subprocess.call(["mkdir", localPath + '{}'.format(curDate)], stdin=False, stdout=False)
    subprocess.call(["mkdir", localPath + '{}/{}'.format(curDate, fn)], stdin=False, stdout=False)
    subprocess.call(["mkdir", localPath + '{}/{}/{}'.format(curDate, fn, subFn)], stdin=False, stdout=False)
    # 续上#2回，注释部分#

    dataPath = ''
    # 读取对应路径
    if featsType == 'hive':
        dataPath = '/user/hive/warehouse/xy_linsanji.db/{}/0*'.format(fn)
        sepStr = '\x01'
    elif featsType == 'hdfs':
        dataPath = '/user/hive/warehouse/xy_app_spark.db/snapshot/feature-regress/{}/xy_wulichuang.{}/{}/part*'.format(
            curDate, fn, subFn)
        # dataPath = '/user/hive/warehouse/xy_jiangyuande.db/scorecard_v2/{}/part*'.format(fn)
    # 计算好坏用户数
    print('==================================== Welcome to Lin\'s world ^-^ ! ====================================')
    startTime = datetime.datetime.timestamp(datetime.datetime.now())
    startTimeStr = datetime.datetime.strftime(datetime.datetime.fromtimestamp(startTime), '%Y-%m-%d %H:%M:%S')
    print('{} #1.  Start base feats calculate...'.format(startTimeStr))
    # oot拆分
    random.seed(2018)
    selectTrainSet = random.sample(range(100), trainSetPercentage)
    # 根据身份证日期来拆分训练测试，因为日期属于均匀分布，有点取巧，待优化
    if diffStrategy == '1':
        dataSplit = sc.textFile(dataPath).repartition(numExecutors). \
            mapPartitions(lambda f: map(lambda x: x.split(sepStr)[0] + '_' +
                                                  ('ootSet' if pd.to_datetime(x.split('_')[2], format='%Y-%m-%d') >=
                                                               pd.to_datetime(ootSplitDate, format='%Y-%m-%d') else
                                                   'trainSet' if len(x.split('_')[0]) != 18 else 'trainSet'
                                                   if int(x.split('_')[0][14:17]) % 99 in selectTrainSet else
                                                   'devSet') + sepStr + sepStr.join(x.split(sepStr)[1:]), f)).persist(
            storageLevel)
    else:
        dataSplit = sc.textFile(dataPath).repartition(numExecutors). \
            mapPartitions(lambda f: map(lambda x: x.split(sepStr)[0] + '_' +
                                                  ('ootSet' if x.split(sepStr)[0].split('_')[-1] == 'oot' else
                                                   'trainSet' if len(x.split('_')[0]) != 18 else 'trainSet'
                                                   if int(x.split('_')[0][14:17]) % 99 in selectTrainSet else
                                                   'devSet') + sepStr + sepStr.join(x.split(sepStr)[1:]), f)).persist(
            storageLevel)
    dataSplit.take(1)
    # save three set
    trainPath = savePath + '{}/{}/{}/trainSet'.format(curDate, fn, subFn)
    devPath = savePath + '{}/{}/{}/devSet'.format(curDate, fn, subFn)
    ootPath = savePath + '{}/{}/{}/ootSet'.format(curDate, fn, subFn)
    testPath = savePath + '{}/{}/{}/testSet'.format(curDate, fn, subFn)
    dataTrain = dataSplit.filter(lambda x: x.split(sepStr)[0].split('_')[-1] == 'trainSet').repartition(
        numExecutors).persist(storageLevel)
    dataTrain.take(1)
    dataTrain.saveAsTextFile(trainPath)
    dataDev = dataSplit.filter(lambda x: x.split(sepStr)[0].split('_')[-1] == 'devSet').persist(storageLevel)
    dataDev.take(1)
    dataDev.saveAsTextFile(devPath)
    dataDev.unpersist()
    dataOot = dataSplit.filter(lambda x: x.split(sepStr)[0].split('_')[-1] == 'ootSet').persist(storageLevel)
    dataOot.take(1)
    dataOot.saveAsTextFile(ootPath)
    dataOot.unpersist()
    dataSplit.unpersist()

    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #1.1 dataTrain has #{}, dataDev has #{}, dataOot has #{} ! Cost {} mins !'.format(endTimeStr,
                                                                                                dataTrain.count(),
                                                                                                dataDev.count(),
                                                                                                dataOot.count(),
                                                                                                costTime))

    # 统计好坏比例
    goodTotal, badTotal = dataTrain.mapPartitions(
        lambda f: map(lambda x: ((float(x.split('_')[3][0]) == 0) * 1, (float(x.split('_')[3][0]) == 1) * 1), f)). \
        reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    # 总用户数
    Total = goodTotal + badTotal
    # 坏用户总占比
    overallRate = badTotal * 1.0 / Total
    # 多线程并行数
    startColNums = 1  # 起始下标
    endColNums = len(dataTrain.take(1)[0].split(sepStr))  # 结束下标
    featureAllList = list(range(startColNums, endColNums))  # 总变量数
    splitedFeaturesPart = SplitFeatures(featureAllList, numLoopParts)
    # 耗时
    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #1.  Finish base feats calculate ! #G is {}, #B is {} ! Cost {} mins !'.format(endTimeStr, goodTotal,
                                                                                             badTotal, costTime))
    # 分不同类别标签循环跑变量筛选
    taskResult = []
    taskQueue = []
    startTime1 = startTime
    print('{} #2.  Start table {} with {} calculate...'.format(endTimeStr, fn, subFn))
    # 续上#2回，注释部分#
    for j in range(len(splitedFeaturesPart)):
        startTime, endTimeStr, costTime = tickClock(startTime)
        print('{} #2.1 Start part #{} calc...'.format(endTimeStr, j))
        # 划分并行集合
        splitedFeatures = SplitFeatures(splitedFeaturesPart[j], numThreads)
        numThreadsNew = len(splitedFeatures)
        # 并行执行
        taskQueue = Queue()
        taskList = []
        for i in range(numThreadsNew):
            currFeatures = splitedFeatures[i]
            paramsDict = [currFeatures, sepStr, maxInterval, maxElements, minIvThreshold]
            t = threading.Thread(target=FeatureRoughSelectQueue, args=[dataTrain, paramsDict])
            taskList.append(t)
            t.start()
        for task in taskList:
            task.join()
        while not taskQueue.empty():
            taskResult = taskResult + taskQueue.get()
        startTime, endTimeStr, costTime = tickClock(startTime)
        print('{} #2.1 Finish part #{} calc ! Cum #{} feats ! Cost {} mins !'.format(endTimeStr, j, len(taskResult),
                                                                                     costTime))
    joblib.dump(taskResult, localPath + '{}/{}/{}/RoughIVSelected.pkl'.format(curDate, fn, subFn))
    # taskResult = joblib.load(localPath + '{}/{}/{}/RoughIVSelected.pkl'.format(curDate, fn, subFn))
    # 续上#2回，注释部分#
    # 进行基于IV值阈值筛选变量
    featureCutOffPointMap = {}
    featureWoeMap = {}
    featureIvMap = {}
    for i in taskResult:
        if i[1][2] > minIvThreshold:
            featureCutOffPointMap.update({fn + '_' + subFn + '_' + str(i[0]): i[1][0]})
            featureIvMap.update({fn + '_' + subFn + '_' + str(i[0]): i[1][2]})
            featureWoeMap.update({fn + '_' + subFn + '_' + str(i[0]): i[1][3]})
    joblib.dump(featureCutOffPointMap, localPath + '{}/{}/{}/featureCutOffPointMap.pkl'.format(curDate, fn, subFn))
    joblib.dump(featureWoeMap, localPath + '{}/{}/{}/featureWoeMap.pkl'.format(curDate, fn, subFn))
    joblib.dump(featureIvMap, localPath + '{}/{}/{}/featureIvMap.pkl'.format(curDate, fn, subFn))
    startTime, endTimeStr, costTime = tickClock(startTime1)
    print('{} #2.  Finish table calc ! IV bigger than {} has #{} feats ! Cost {} mins !'.
          format(endTimeStr, minIvThreshold, len(featureIvMap), costTime))
    print('{} #3.  Start Del higher than {} Corr feats...'.format(endTimeStr, rohThreshold))
    numSaveRepartitions = numExecutors * numThreads
    dfCacheWoe = dataTrain.repartition(numSaveRepartitions). \
        mapPartitions(lambda f: map(lambda x: [
        float(AssignGroupWOE(float(x.split(sepStr)[int(real_curr.split('_')[-1])]), featureCutOffPointMap[real_curr],
                             featureWoeMap[real_curr]))
        for real_curr in list(featureIvMap)], f)).persist(storageLevel)
    # 单文件相关性处理，剔除高相关性低iv变量
    removedVar = []
    restVars = list(featureIvMap)
    loop_circle = len(restVars)
    pearsonCorr = Statistics.corr(dfCacheWoe, method='pearson')
    for i in range(loop_circle - 1):
        for j in range(i + 1, loop_circle):
            if j > i:
                if abs(pearsonCorr[i, j]) >= rohThreshold \
                        and restVars[i] not in removedVar \
                        and restVars[j] not in removedVar:
                    # 选择IV高的变量保留,剔除另一个变量
                    if featureIvMap[restVars[i]] > featureIvMap[restVars[j]]:
                        removedVar.append(restVars[j])
                    else:
                        removedVar.append(restVars[i])
    # 最终保留变量
    keepFeaturesFinal = [i for i in restVars if i not in removedVar]
    joblib.dump(keepFeaturesFinal, localPath + '{}/{}/{}/keepFeaturesFinal.pkl'.format(curDate, fn, subFn))
    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #3.  Finish Del high Corr feats ! Keep #{} ! Cost {} mins !'.format(endTimeStr, len(keepFeaturesFinal),
                                                                                  costTime))
    print('{} #4.  Start store feats ...'.format(endTimeStr))
    # save trainSet selected vars
    trainSetWoe = saveWoe(trainPath, numSaveRepartitions, keepFeaturesFinal, featureCutOffPointMap, featureWoeMap)
    trainSetWoe.saveAsTextFile(savePath + '{}/{}/{}/trainSetWoe'.format(curDate, fn, subFn))
    # raw trainSet
    trainSetRaw = saveRaw(trainPath, numSaveRepartitions, keepFeaturesFinal)
    trainSetRaw.saveAsTextFile(savePath + '{}/{}/{}/trainSetRaw'.format(curDate, fn, subFn))
    # # save devSet selected vars
    devSetWoe = saveWoe(devPath, numSaveRepartitions, keepFeaturesFinal, featureCutOffPointMap, featureWoeMap)
    devSetWoe.saveAsTextFile(savePath + '{}/{}/{}/devSetWoe'.format(curDate, fn, subFn))
    # # raw devSet
    devSetRaw = saveRaw(devPath, numSaveRepartitions, keepFeaturesFinal)
    devSetRaw.saveAsTextFile(savePath + '{}/{}/{}/devSetRaw'.format(curDate, fn, subFn))
    # # save ootSet selected vars
    ootSetWoe = saveWoe(ootPath, numSaveRepartitions, keepFeaturesFinal, featureCutOffPointMap, featureWoeMap)
    ootSetWoe.saveAsTextFile(savePath + '{}/{}/{}/ootSetWoe'.format(curDate, fn, subFn))
    # # raw ootSet
    ootSetRaw = saveRaw(ootPath, numSaveRepartitions, keepFeaturesFinal)
    ootSetRaw.saveAsTextFile(savePath + '{}/{}/{}/ootSetRaw'.format(curDate, fn, subFn))
    startTime, endTimeStr, costTime = tickClock(startTime)
    print('{} #4.  Finish store feats ! Cost {} mins !'.format(endTimeStr, costTime))
    print('==================================== Have Fun ! See ya next time ^-^ ! ====================================')

