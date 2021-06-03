import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import dirUtil 
from highDimLearning import vgg2feat
from sklearn.metrics.pairwise import euclidean_distances
import os
import random 
from scipy.linalg import null_space
from sklearn.neighbors.kde import KernelDensity
from sklearn.decomposition import PCA
from highDimLearningX import estModel2
from clusteringXX import clusterData
import copy
from sklearn.metrics import pairwise_distances

def density_scoring(trainingFeat, sameFeat, diffFeat, bandwidth=0.001):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(trainingFeat)
    sameScore = kde.score_samples(sameFeat)
    diffScore = kde.score_samples(diffFeat)

    comb = -1.0 * np.concatenate([sameScore, diffScore])
    
    ind = np.argsort(comb)
    est = np.zeros(ind.size)
    gt  = np.zeros(ind.size)
    thres = sameFeat.shape[0]
    runner = 0
    for i in ind:
        if runner<thres:
            gt[runner] = 1
            est[i] = 1
            runner = runner +1
            
    score = 1 - np.sum(np.absolute(est-gt))/gt.size
    return score

def nn_dist_scoring(trainingFeat, sameFeat, diffFeat):
    dd = np.min(pairwise_distances(trainingFeat, sameFeat), axis=0)
    DD = np.min(pairwise_distances(trainingFeat, diffFeat), axis=0)
    comb = np.concatenate((dd, DD))
    ind = np.argsort(comb)
    _est = np.zeros(ind.shape[0])
    _gt = np.zeros(ind.shape[0])
    thres = dd.shape[0]
    runner = 0
    for i in ind:
        if runner < thres:
            _gt[runner] = 1
            _est[i] = 1
            runner += 1
        
    score = 1 - np.sum(np.abs(_est-_gt)) / ind.shape[0]
    return score

def hyper_iterative_fit(data, numIter=5, save_transforms=False, folder='./', cid=0):
    pca = PCA(n_components=numIter)
    pca.fit(data)
    m = pca.components_
    sig = np.mean(np.matmul(data, np.transpose(m)), axis=0)
    W, Sig = estModel2(data, np.transpose(m), sig, numShells=numIter, weight=1.0)

    if save_transforms:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filename = folder + 'hyperfit_' + str(cid) + '.npz'
        np.savez(filename, ns=None, w=W, sig=Sig)

    return W, Sig

def cluster_variance(data):
    if data.shape[0] <= 10000:
        s = int(data.shape[0])
    else:
        s = 10000
        idx = random.sample(range(data.shape[0]), 10000)
        data = data[idx]
    
    affinity = np.zeros((s, s))
    affinity = euclidean_distances(data, data) # / data.shape[1]
    arr = np.zeros(int((s-1)*s/2))
    runner = 0
    for i in range(s):
        for j in range(s):
            if i > j:
                arr[runner] = affinity[i, j]
                runner += 1
            else:
                break
    return np.std(arr)


def iterative_cluster(data, save_transforms=False, folder="./", THRESHOLD=0.02, n_branch=2):
    cur_data = copy.deepcopy(data)
    tot_dim = data.shape[1]
    transform = np.eye(tot_dim, dtype=float)
    cur_std = cluster_variance(cur_data)
    final_feats = np.zeros((cur_data.shape[0], n_branch))
    new_feats = np.zeros((cur_data.shape[0], n_branch))
    cur_feats = np.zeros((cur_data.shape[0], n_branch))
    print("Original std: ", cur_std)
    i = 0
    
    transform_ls = []
    while True:
        
        if i == 0:
            w, sig, sigA, label, new_feats, labelK = clusterData(cur_data, n_branch)
            cur_feats = new_feats
            final_feats = new_feats
            final_data = cur_data
            transform_ls.append((transform, w, sig))
            transform = null_space(np.transpose(w))
            i += 1
        else:
            if not i == 1:
                cur_feats = np.column_stack((cur_feats, new_feats))
            cur_data = np.matmul(cur_data, transform)
            new_std = cluster_variance(cur_data)
            print("i = ", i, ": ", new_std)
            if (cur_std-new_std)/cur_std < THRESHOLD:
                print("Num of splits = ", i-1, " - clustering fit training completed.")
                break
            else:
                cur_std = new_std
                final_data = cur_data
                final_feats = cur_feats    
                w, sig, sigA, label, new_feats, labelK = clusterData(cur_data, n_branch)
                transform_ls.append((transform, w, sig))
                transform = null_space(np.transpose(w))
                i += 1

    if save_transforms:
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i in range(len(transform_ls)):
            ns, w, sig = transform_ls[i]
            filename = folder + str(i) + '.npz'
            np.savez(filename, ns=ns, w=w, sig=sig)
    
    return transform_ls, i-1, final_data, final_feats

def unit_norm(matrix, ax=1):
    return matrix / np.linalg.norm(matrix, axis=ax, keepdims=True)