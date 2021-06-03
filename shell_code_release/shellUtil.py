import numpy as np
from random import sample 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def eval_(estLabel, score, gtLabel, verboise=True):

    numClass = np.max(gtLabel)+1
    acc = np.mean(estLabel==gtLabel)
    meanAveragePrecision = average_precision_score(estLabel==gtLabel, np.max(score, axis=1))


    topX = np.zeros(numClass)
    roc = np.zeros(numClass)
    for i in range(numClass):
        roc[i] = roc_auc_score(gtLabel==i, score[:,i])
        ind = np.argsort(-score[:,i])
        #topX[i] = sum(gtLabel[ind[:1000]]== i)

    if verboise:
        print('Mean Accuracy:', acc)
        print('MAP:', meanAveragePrecision)
        print('AUROC:', np.mean(roc))
        #print('ROC:', roc)
        #print('TopX:', topX)


    return acc, meanAveragePrecision, roc


def fitToList(feat, gt):
    # converts an array of features into a list based on their groundTruth Class
    featList = []
    numClass = np.max(gt)+1
    for i in range(numClass):
        featList.append(feat[gt==i])
    return featList


def normIt(data, m=None):
    nData = data.copy()
    #nData = data/np.linalg.norm(data, axis =1, keepdims=True)
    if m is None:
        m = np.mean(nData, axis =0, keepdims=True)
    nData = nData - m
    nData = nData/np.linalg.norm(nData, axis =1, keepdims=True)
    
    return nData, m


def sorted_neighbors_of_i(m_all, i):
    neighbors = np.zeros(m_all.shape[0])
    for j in range(m_all.shape[0]):
        neighbors[j] = np.linalg.norm(m_all[i,:]-m_all[j,:])
    return neighbors, np.argsort(neighbors)
   
def classAndOutliers(gt, trueClass, outlierPercentage, outlierClassList=[]):
    numClass = np.max(gt)+1
    if len(outlierClassList) == 0:
        for i in range(numClass):
            if not i == trueClass:
                outlierClassList.append(i)
    
    finalMask = np.zeros(gt.size, dtype=bool)

    outlierPerClass = int(np.sum(gt==trueClass)*outlierPercentage/len(outlierClassList))


    for c in outlierClassList:
        mask = gt == c
        ind = np.where(mask)[0]
        finalMask[sample(list(ind),outlierPerClass)] = 1
    finalMask[gt==trueClass] = 1

    return finalMask




