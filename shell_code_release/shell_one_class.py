import numpy as np
from sklearn.svm import OneClassSVM
from dataLoader import importData
from shellUtil import normIt
from shellUtil import eval_
import timeit




def evalOneClassShell(testFeat, testGt, trainFeat, trainGt, verboise=True, withShellNorm=True, norm_vec=None):
    
    if withShellNorm:
        trainFeat, _ = normIt(trainFeat, norm_vec)
        testFeat, _ = normIt(testFeat, norm_vec)

    featList = []
    numClass = np.max(trainGt)+1
    for i in range(numClass):
        featList.append(trainFeat[trainGt==i])

    trainTime = 0
    testTime = 0
    scores = np.zeros([testFeat.shape[0], numClass])
    for i in range(numClass):       
        sOCM = OneClassMean()
        
        # training
        start = timeit.default_timer()    
        sOCM.train(featList[i])
        stop = timeit.default_timer()   
        trainTime = trainTime + stop-start 

        # testing
        start = timeit.default_timer()    
        scores[:,i] = sOCM.score(testFeat)
        stop = timeit.default_timer()   
        testTime = testTime + stop-start 

    trainTime = trainTime/numClass
    testTime = testTime/numClass
    if verboise:
        print('Train time per class: ', trainTime)
        print('Test time per class: ', testTime)

    labelEst = np.argmax(scores, axis=1)

    meanEST, mapEST, rocEST = eval_(labelEst, scores, testGt, verboise)
    return meanEST, mapEST, rocEST



# assumes data has been normalized
class OneClassMean:

    def __initi__(self):
        self.classifer = None

    def train(self, feat):
        self.classifer = ocMean(feat)

    def score(self, feat): 
        # smaller scores are better, muliply - to reverse that
        score = ocMeanScore(feat, self.classifer)
        return -score    



def ocMean(feat):
    m_ = np.mean(feat, axis=0, keepdims=True)
    d = feat - m_
    d = np.linalg.norm(d, axis=1)
    
    model ={'clusMean': m_, 
            'numInstance': feat.shape[0], 
            'noiseMean': np.median(d), 
            'noiseStd':np.median(np.absolute(d-np.mean(d))), 
            'mean_norm': 0}
    return model

def ocMeanScore(feat, model):
    feat_ = feat.copy()  
    feat_ = feat_ -  model['clusMean']
    feat_ = np.linalg.norm(feat_, axis=1)
    ss = (feat_ - model['noiseMean'])/model['noiseStd']
    #ss[ss<0] = 0    
    return ss


if __name__ == "__main__":

    trainFeat, trainGt, testFeat, testGt, _ = importData(4)
    mVec = np.mean(testFeat, axis=0, keepdims=True) # estimate mean of test features

    print('\nShell Learning  with Raw Features\n')
    evalOneClassShell(testFeat, testGt, trainFeat, trainGt, verboise=True, 
                    withShellNorm=False);


    print('\nShell Learning  with Shell Normalization\n')
    evalOneClassShell(testFeat, testGt, trainFeat, trainGt, verboise=True, 
                    withShellNorm=True, norm_vec = mVec,);

    print('\nShell normalization significantly increases accuracy.')
    print('Shell learning significantly faster than ocsvm (see ocsvm_basic).')
    print('Shell normalization uses the mean of the test-data; this differs from traditional normalization that uses the mean of the training data).')
