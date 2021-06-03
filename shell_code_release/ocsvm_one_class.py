import numpy as np
from sklearn.svm import OneClassSVM
from shellUtil import normIt
from shellUtil import sorted_neighbors_of_i
from shellUtil import eval_
from dataLoader import importData
import timeit




def evalOneClassSVM(testFeat, testGt, trainFeat, trainGt, verboise=True, withShellNorm=True,
                    norm_vec=None, divideByNum=False):

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
        svm = OneClassSVM(gamma='auto')
        # training
        start = timeit.default_timer()    
        svm.fit(featList[i])
        stop = timeit.default_timer()    
        trainTime = trainTime + stop-start
        
        
        start = timeit.default_timer()    
        if divideByNum:
            scores[:,i] = svm.score_samples(testFeat)/featList[i].shape[0]
        else:
            scores[:,i] = svm.score_samples(testFeat)
        stop = timeit.default_timer()    
        testTime = testTime + stop-start
    trainTime = trainTime/numClass
    testTime = testTime/numClass
    if verboise:
        print('Train Time: ', trainTime)
        print('Test Time: ', testTime)

    labelEst = np.argmax(scores, axis=1)

    meanEST, mapEST, rocEST = eval_(labelEst, scores, testGt, verboise)
    return meanEST, mapEST, rocEST



if __name__ == "__main__":

    trainFeat, trainGt, testFeat, testGt, _ = importData(1)
    mVec = np.mean(testFeat, axis=0, keepdims=True) # estimate mean of test features



    print('\nTraditional One Class SVM (no normalization)\n')
    evalOneClassSVM(testFeat, testGt, trainFeat, trainGt, 
                    withShellNorm=False, 
                    verboise=True, divideByNum=False);

    print('\nOne Class SVM with Shell Normalization and scaling by number of features\n')
    evalOneClassSVM(testFeat, testGt, trainFeat, trainGt, 
                    withShellNorm=True, norm_vec = mVec,
                    verboise=True, divideByNum=True);


    print('\nShell normalization significantly increases accuracy.\n')



