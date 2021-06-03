import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clusteringXXX import cluster_data
from clusteringXXX import cluster_project
from clusteringXXX import del_dimensions
from sklearn.metrics import average_precision_score
from scipy.optimize import linear_sum_assignment


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum()/Y_pred.size, w, row_ind, col_ind


def converge_labels2(y_pred, y, estimated_classes):

    _, aff, _, _ = cluster_acc(y_pred, y)
    a = aff[:estimated_classes]
    map_ = np.argmax(a, axis =0)

    y_new = -np.ones(y.size, dtype=int)

    for i in range(map_.size):
        mask = y ==i
        y_new[mask] = map_[i]

    return y_new

def converge_labels(Y_pred, Y):
    _, _, row_ind, col_ind = cluster_acc(Y_pred, Y)
    new_labels = -np.ones(Y.size, dtype=int)
    for i in range(row_ind.size):
        mask = Y_pred == row_ind[i]
        new_labels[mask] = col_ind[i]
    return new_labels


def eval_labels(pred_labels, gt, score):
    predicted_classes = np.max(pred_labels)+1
    new_gt = converge_labels2(pred_labels, gt.astype(int), predicted_classes)

    purity = np.sum((pred_labels == new_gt))/gt.size
    mean_average_precision = average_precision_score(
        pred_labels == new_gt, score)

    return purity, mean_average_precision


def eval_ours(feat, num_clus, gt, add_dim_per_class=100):
    scaler = StandardScaler(copy=True, with_mean=True,
                            with_std=False).fit(feat)
    new_feat = scaler.transform(feat, copy=True)
    new_feat = new_feat/np.linalg.norm(new_feat, keepdims=True, axis=1)

    w, sig, labelK = cluster_data(
        new_feat, numClass=num_clus, add_dim_per_class=add_dim_per_class)

    hyper_label, hyper_feat = cluster_project(new_feat, w, sig, num_clus)

    hyper_feat, hyper_label, mask = del_dimensions(hyper_feat)

    purity, mean_average_precision = eval_labels(
        hyper_label, gt, np.max(hyper_feat, axis=1))

    return purity, mean_average_precision, hyper_label, [w, sig, mask]

def eval_kmeans(feat, num_clus, gt):
    scaler = StandardScaler(copy=True, with_mean=True,
                            with_std=False).fit(feat)
    new_feat = scaler.transform(feat, copy=True)
    new_feat = new_feat/np.linalg.norm(new_feat, keepdims=True, axis=1)

    kmeans = KMeans(n_clusters=num_clus, random_state=0).fit(new_feat)
    label_k = kmeans.predict(new_feat)
    score = np.min(kmeans.transform(new_feat), axis=1)

    purity, mean_average_precision = eval_labels(
        label_k, gt, -score)


    return purity, mean_average_precision, label_k, kmeans
