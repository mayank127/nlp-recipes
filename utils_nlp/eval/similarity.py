import faiss
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np
import collections

class NeighborSearch():
    def __init__(self, search_matrix, is_gpu_available=True):
        self.search_matrix = search_matrix
        self.dimension = search_matrix.shape

        if is_gpu_available:
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatIP(res, self.dimension[1])
        else:
            self.index = faiss.IndexFlatIP(self.dimension[1])

        self.index.add(search_matrix)

    # given a tensor or a batch of tensor returns distance and index to closest target neighbours
    def NN(self, v, knn=1):
        cv = v
        if len(cv.shape) == 1:
            cv = cv.reshape(1, cv.shape[0])
        D, I = self.index.search(cv, knn)
        return D, I


def get_cosine_ccuracy(source_emb, target_emb, source_to_target, batch_size=500):
    dtype = 'float32'
    source_emb = source_emb.astype(dtype)
    target_emb = target_emb.astype(dtype)

    source_keys = list(source_to_target.keys())
    source_keys_len = len(source_keys)

    nbrhood_src_index = np.zeros((source_keys_len, 1))

    search_space = NeighborSearch(target_emb)

    for i in tqdm(range(0, source_keys_len, batch_size)):
        j = min(i + batch_size, source_keys_len)
        _, nbrhood_src_index[i:j,:] = search_space.NN(source_emb[source_keys[i:j]], knn=1)

    accuracy = np.mean([1 if nbrhood_src_index[i][0] in source_to_target[source_keys[i]] else 0 for i in range(source_keys_len)])
    return accuracy


def _get_neighborhood(source_emb, target_emb, vectors_to_search, knn, batch_size):
    nbrhood = np.zeros(source_emb.shape[0])

    search_space = NeighborSearch(target_emb)

    for i in tqdm(range(0, len(vectors_to_search), batch_size)):
        j = min(i + batch_size, len(vectors_to_search))
        similarities, _ = search_space.NN(source_emb[vectors_to_search[i:j]], knn=knn)
        nbrhood[vectors_to_search[i:j]] = np.mean(similarities, axis=1)

    return nbrhood


def get_csls_accuracy(source_emb, target_emb, source_to_target, knn=10, batch_size=500):
    dtype = 'float32'
    source_emb = source_emb.astype(dtype)
    target_emb = target_emb.astype(dtype)

    source_keys = list(source_to_target.keys())
    source_keys_len = len(source_keys)

    translation = collections.defaultdict(int)

    # neighborhood of source indexes in target
    nbrhood_src = _get_neighborhood(source_emb, target_emb, source_keys, knn, batch_size)

    # neightborhood of all target indexes in source
    nbrhood_trg = _get_neighborhood(target_emb, source_emb, list(range(target_emb.shape[0])), knn, batch_size)

    for i in tqdm(range(0, source_keys_len, batch_size)):
        j = min(i + batch_size, source_keys_len)
        similarities = source_emb[source_keys[i:j]].dot(target_emb.T)

        # 2 * cosine - sigma(nbr_src) - sigma(nbr_trg)
        similarities = np.transpose(np.transpose(2 * similarities) - nbrhood_src[source_keys[i:j]]) - nbrhood_trg
        arg_nearest = similarities.argmax(axis=1).tolist()
        similarities = np.argsort((similarities),axis=1)

        for k in range(j-i):
            translation[source_keys[i + k]] = arg_nearest[k]

    accuracy = np.mean([1 if translation[i] in source_to_target[i] else 0 for i in source_keys])
    return accuracy