import numpy as np
from sklearn.metrics import roc_auc_score

from dataLoader import importData
from shellUtil import classAndOutliers
from shellUtil import normIt

def robustMean(featTrain, globalMean, thres=2, numIter=10):
    
    feat, _ = normIt(featTrain, globalMean)
    m_, var, eSig = estShell(feat)
    err = projectMean(feat, m_, var)
    mask = err<eSig*thres
    meanInlier = np.mean(featTrain[mask,:], axis=0)
    meanOutlier = np.mean(featTrain[~mask,:], axis=0)
    globalMean = (meanInlier + meanOutlier)/2


    
    for i in range(numIter):
        feat, _ = normIt(featTrain, globalMean)
        m_, var, eSig = estShell(feat[mask])
        err = projectMean(feat, m_, var)
        mask = err<eSig*thres

        meanInlier = np.mean(featTrain[mask,:], axis=0)
        meanOutlier = np.mean(featTrain[~mask,:], axis=0)
        globalMean = (meanInlier + meanOutlier)/2
        #newMean = meanOutlier


    return err, mask


def robustMean_simple(featTrain, globalMean, thres=3, numIter=3):

    feat, _ = normIt(featTrain, globalMean)
    m_, var, eSig = estShell(feat)
    err = projectMean(feat, m_, var)
    mask = err<eSig*thres

    for i in range(numIter):
        m_, var, eSig = estShell(feat[mask])
        err = projectMean(feat, m_, var)
        mask = err<eSig*thres
    return err, mask

def estShell(data):
    m_ = np.mean(data, axis=0, keepdims=True)
    d = np.linalg.norm(data - m_, axis=1)
    var = np.mean(d)

    err = np.absolute(d-var)
    MAD =  np.median(err)
    eSig = 1.4826*MAD

    return m_, var, eSig

def projectMean(data, m, var):
    d = np.linalg.norm(data - m, axis=1)
    err = d-var
    return err



if __name__ == "__main__":

    dataSet = 1
    outlierPercent = 0.5


    print('Outlier percentage:', outlierPercent)
    trainFeat, trainGt, testFeat, testGt, _ = importData(dataSet)
    allFeat = np.concatenate([trainFeat, testFeat], axis=0)
    gtAll = np.concatenate([trainGt, testGt])

    numClass = np.max(gtAll) + 1
    trueClass = 0
    dataMask = classAndOutliers(gtAll, trueClass, outlierPercent, outlierClassList=[])
    f_ = allFeat[dataMask]
    gt_ = (gtAll == trueClass)[dataMask]


    # new_m = np.mean(allFeat, axis=0, keepdims=True)
    # err, mask = robustMean(f_, new_m, thres=2, numIter=10)
    # auroc = roc_auc_score(gt_, -err)
    # print('\nShell auroc, normalization with true test mean:', auroc)

    print('\nShell Basic, normalized with Flickr11k mean')
    new_m = np.load('data/flickr11k_mean.npy')
    err, mask = robustMean_simple(f_, new_m, thres=2, numIter=10)
    auroc = roc_auc_score(gt_, -err)
    print('auroc:', auroc)

    print('\nShell Re-normalized, initialized with Flickr11k mean')
    new_m = np.load('data/flickr11k_mean.npy')
    err, mask = robustMean(f_, new_m, thres=2, numIter=10)
    auroc = roc_auc_score(gt_, -err)
    print('auroc:', auroc)

    print('\nRe-normalization refines the global-mean used for normalization.')
