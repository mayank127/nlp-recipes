import numpy as np
import tensorflow as tf
import scipy.linalg

from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product, PositiveDefinite
from pymanopt.solvers import ConjugateGradient
from pymanopt.tools.autodiff import TensorflowBackend


def get_geomm_mapping(lang_graph, emb_vectors, low_rank=300, lr=1e-3, seed=0,
    tf_dtype=tf.float32, maxiter=100, maxtime=20000, verbosity=1):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    dimension = -1
    for key in emb_vectors:
        if dimension == -1:
            dimension = emb_vectors[key].vectors.shape[1]
        else:
            assert emb_vectors[key].vectors.shape[1] == dimension,\
                "All embedding dimensions should be same"

    assert low_rank <= dimension,\
        "Low rank should be smaller than embedding dimensions"

    uniq_langs = set()
    for edge in lang_graph:
        uniq_langs.add(edge.src_label)
        uniq_langs.add(edge.trg_label)
    uniq_langs = list(uniq_langs)
    indexes = {lang:i+1 for i, lang in enumerate(uniq_langs)}

    B_tensor = tf.Variable(tf.placeholder(tf_dtype), validate_shape=False)
    B_manifold = PositiveDefinite(low_rank)

    all_tensors = [B_tensor]
    all_manifolds = [B_manifold]

    for lang in uniq_langs:
        all_tensors.append(tf.Variable(tf.placeholder(tf_dtype), validate_shape=False))
        all_manifolds.append(Stiefel(dimension, low_rank))

    cost =  0.5*lr*(tf.math.reduce_sum(B_tensor**2))

    for edge in lang_graph:
        src_label = edge.src_label
        trg_label = edge.trg_label
        mapping = edge.mapping

        Kx = emb_vectors[src_label].vectors[mapping.uniq_src]
        Kz = emb_vectors[trg_label].vectors[mapping.uniq_trg]
        A = mapping.A

        tf_XtAZ = tf.convert_to_tensor(Kx.T.dot(A.dot(Kz)), dtype=tf_dtype)
        tf_XtX = tf.convert_to_tensor(Kx.T.dot(Kx), dtype=tf_dtype)
        tf_ZtZ = tf.convert_to_tensor(Kz.T.dot(Kz), dtype=tf_dtype)

        U = all_tensors[indexes[src_label]]
        V = all_tensors[indexes[trg_label]]
        B = all_tensors[0]
        W = tf.matmul(tf.matmul(U, B), V, transpose_b=True)

        wtxtxw = tf.matmul(W, tf.matmul(tf_XtX, W), transpose_a=True)
        wtxtxwztz = tf.matmul(wtxtxw, tf_ZtZ)

        cost += tf.linalg.trace(wtxtxwztz)
        cost += -2 * tf.math.reduce_sum(W * tf_XtAZ)

    solver = ConjugateGradient(maxiter=maxiter, maxtime=maxtime)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        problem = Problem(manifold=Product(all_manifolds), cost=cost, arg=all_tensors, verbosity=verbosity)

        # hack to use correct session in tflow
        problem._backend = TensorflowBackend()
        problem._backend._session.close()
        problem._backend._session = sess

        result = solver.solve(problem)

    B_result = result[0]
    B_result = (B_result + B_result.T)/2
    B_root = scipy.linalg.sqrtm(B_result)

    U_results = {lang:result[indexes[lang]] for lang in  uniq_langs}
    return B_root, U_results






