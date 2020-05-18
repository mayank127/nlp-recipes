import numpy as np
import collections

from utils_nlp.common.matrix_util import length_normalize
from utils_nlp.eval.similarity import get_cosine_ccuracy, get_csls_accuracy

class LangMapping:
    def __init__(self, uniq_src, uniq_trg, A):
        self.uniq_src = uniq_src
        self.uniq_trg = uniq_trg
        self.A = A

class LangEdge:
    def __init__(self, src_label, trg_label, mapping):
        self.src_label = src_label
        self.trg_label = trg_label
        self.mapping = mapping

def _get_new_Xw(Kx, U, B_root, normalize=True):
    Xw = Kx.dot(U).dot(B_root)
    if normalize:
        return length_normalize(Xw)
    else:
        return Xw

def evaluate_lang_pair(vector_src, vector_trg, U_src, U_trg, B_root, test_dictionary):
    xw = _get_new_Xw(vector_src, U_src, B_root)
    zw = _get_new_Xw(vector_trg, U_trg, B_root)

    cosine_acc = get_cosine_ccuracy(xw, zw, test_dictionary)
    csls_acc = get_csls_accuracy(xw, zw, test_dictionary)
    return {"Cosine Accuracy": cosine_acc, "CSLS Accuracy": csls_acc}

def create_train_mapping(pair_dataframe, src_emb, trg_emb):
    src_indices = []
    trg_indices = []
    for _, row in pair_dataframe.iterrows():
        src_word = row.src_word.lower()
        trg_word = row.trg_word.lower()
        try:
            src_index = src_emb.vocab[src_word].index
            trg_index = trg_emb.vocab[trg_word].index
            src_indices.append(src_index)
            trg_indices.append(trg_index)
        except KeyError as e:
            pass
    x_count = len(set(src_indices))
    z_count = len(set(trg_indices))
    A = np.zeros((x_count,z_count))

    # Creating dictionary matrix from training set
    map_dict_src={}
    map_dict_trg={}
    I=0
    uniq_src=[]
    uniq_trg=[]
    for i in range(len(src_indices)):
        if src_indices[i] not in map_dict_src.keys():
            map_dict_src[src_indices[i]]=I
            I+=1
            uniq_src.append(src_indices[i])
    J=0
    for j in range(len(trg_indices)):
        if trg_indices[j] not in map_dict_trg.keys():
            map_dict_trg[trg_indices[j]]=J
            J+=1
            uniq_trg.append(trg_indices[j])

    for i in range(len(src_indices)):
        A[map_dict_src[src_indices[i]],map_dict_trg[trg_indices[i]]]=1
        
    return LangMapping(uniq_src, uniq_trg, A)

def create_dictionary(pair_dataframe, src_emb, trg_emb):
    src2trg = collections.defaultdict(set)
    for _, row in pair_dataframe.iterrows():
        src_word = row.src_word.lower()
        trg_word = row.trg_word.lower()
        try:
            src_index = src_emb.vocab[src_word].index
            trg_index = trg_emb.vocab[trg_word].index
            src2trg[src_index].add(trg_index)
        except KeyError as e:
            pass
    return src2trg