from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import numpy as np

import dirUtil 
from highDimLearning import estModel
from highDimLearning import estModel2
from highDimLearning import vgg2feat

import tensorflow as tf

#model = VGG16(weights='imagenet', include_top=False)
#model.summary()

from sklearn.metrics.pairwise import euclidean_distances





def label(feat, w, sig):

    numClass = len(w)
    score = np.zeros([feat.shape[0], numClass])
    for i in range(numClass):
        ww = w[i]
        sigg = np.reshape(sig[i], [sig[i].size, 1])
        score[:, i:i+1] = np.matmul(feat, ww)-sigg
    labels = np.argmax(score, axis=1)
    return labels, score

    

def feat2labels(feat):
    numClass = len(feat)
    numDim = feat[0].shape[1]
    numPts = 0
    for i in range(numClass):
        numPts = numPts + feat[i].shape[0]

    allFeat = np.zeros([numPts, numDim])
    allLabels = np.zeros(numPts, dtype = int)
    cur = 0
    for i in range(numClass):
        allFeat[cur:cur+feat[i].shape[0],:] = feat[i]
        allLabels[cur:cur+feat[i].shape[0]] = i
        cur = cur + feat[i].shape[0]
    return allFeat, allLabels
        

trainingNum = 1000


folder = '/media/daniel/D/code/keras-tutorial/animals/dogs/'

feats = dirUtil.dir2vgg16(folder)
feat1 = vgg2feat(feats)
m = np.mean(feat1, axis=0, keepdims=True)
sig = np.mean(np.matmul(feat1, np.transpose(m)))
w1, sig1 = estModel(feat1[:trainingNum, :], np.transpose(m), sig, 1.0)
#w1, sig1 = estModel2(feat1, np.transpose(m), sig, 1, 1)


folder2 = '/media/daniel/D/code/keras-tutorial/animals/panda/'

tf.reset_default_graph()
K.clear_session()

feats = dirUtil.dir2vgg16(folder2)
feat2 = vgg2feat(feats)
m = np.mean(feat2, axis=0, keepdims=True)
sig = np.mean(np.matmul(feat2, np.transpose(m)))
w2, sig2 = estModel(feat2[:trainingNum, :], np.transpose(m), sig, 1.0)


allFeat, gt = feat2labels([feat1, feat2])

########### display

# a = np.matmul(allFeat,w1)-sig1
# b = np.matmul(allFeat,w2)-sig2

# c = np.concatenate([a,b], axis =1)



# from matplotlib import pyplot as plt
# plt.scatter(c[:feat1.shape[0], 0], c[:feat1.shape[0], 1], color='red', alpha = 0.1)
# plt.scatter(c[feat1.shape[0]:-1, 0], c[feat1.shape[0]:-1,1], color='blue', alpha =0.1)
# plt.plot(np.arange(-.5, 1., 0.5), np.arange(-.5, 1., 0.5),  color='black')
# plt.show()

#cv2.waitKey()
##########################


from sklearn.manifold import TSNE

ts1 = TSNE(n_components=1).fit(feat1)
ts2 = TSNE(n_components=1).fit(feat2)

c = np.concatenate([ts1.fit_transform(allFeat), ts2.fit_transform(allFeat)], axis=1)

from matplotlib import pyplot as plt
plt.scatter(c[:feat1.shape[0], 0], c[:feat1.shape[0], 1], color='red', alpha = 0.1)
plt.scatter(c[feat1.shape[0]:-1, 0], c[feat1.shape[0]:-1,1], color='blue', alpha =0.1)
plt.plot(np.arange(-.5, 1., 0.5), np.arange(-.5, 1., 0.5),  color='black')
plt.show()
