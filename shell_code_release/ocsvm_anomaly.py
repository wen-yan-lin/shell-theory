import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

from dataLoader import importData
from shellUtil import classAndOutliers
from shellUtil import normIt




if __name__ == "__main__":

    dataSet = 1
    outlierPercent = 0.5


    print('outlier percentage:', outlierPercent)
    trainFeat, trainGt, testFeat, testGt, _ = importData(dataSet)
    allFeat = np.concatenate([trainFeat, testFeat], axis=0)
    gtAll = np.concatenate([trainGt, testGt])

    numClass = np.max(gtAll) + 1
    trueClass = 0
    dataMask = classAndOutliers(gtAll, trueClass, outlierPercent, outlierClassList=[])
    f_ = allFeat[dataMask]
    gt_ = (gtAll == trueClass)[dataMask]

    print('\nOCSVM, normalized with Flickr11k')
    new_m = np.load('data/flickr11k_mean.npy')
    f, _ = normIt(f_, new_m)
    svm = OneClassSVM(gamma='auto')
    svm.fit(f)
    s = svm.score_samples(f)
    auroc = roc_auc_score(gt_, s)
    print('auroc: ', auroc)