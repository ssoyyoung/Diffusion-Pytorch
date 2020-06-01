#!/usr/bin/env python
# -*- coding: utf-8 -*-

" diffusion module "

import os
import time
import numpy as np
import joblib
from joblib import Parallel, delayed
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from tqdm import tqdm
from knn import KNN, ANN


trunc_ids = None
trunc_init = None
lap_alpha = None


def get_offline_result(i):
    ids = trunc_ids[i]
    trunc_lap = lap_alpha[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=20)
    return scores


def cache(filename):
    """Decorator to cache results
    """
    def decorator(func):
        def wrapper(*args, **kw):
            self = args[0]
            path = os.path.join(self.cache_dir, filename)
            time0 = time.time()
            if os.path.exists(path):
                result = joblib.load(path)
                cost = time.time() - time0
                print('[cache] loading {} costs {:.2f}s'.format(path, cost))
                return result
            result = func(*args, **kw)
            cost = time.time() - time0
            print('[cache] obtaining {} costs {:.2f}s'.format(path, cost))
            joblib.dump(result, path)
            return result
        return wrapper
    return decorator


class Diffusion(object):
    """Diffusion class
    """
    def __init__(self, features, cache_dir):
        # features : query와 gallery feature vector를 위 아래로 합침
        self.features = features
        # 총 개수 확인
        self.N = len(self.features)
        self.cache_dir = cache_dir

        # 10만개 이상의 데이터에 대해서는 ann 수행
        # use ANN for large datasets
        self.use_ann = self.N >= 100000
        if self.use_ann:
            print("use ann")
            self.ann = ANN(self.features, method='cosine')

        # 10만개 이하의 데이터에 대해서는 knn 수행
        self.knn = KNN(self.features, method='cosine')
        print(self.knn)

        # 10만개 이상의 데이터가 INPUT으로 들어갈 경우, 특정 벡터값들이 NAN 인 경우가 있음
        # 이에 해당하는 해결방안 고려(0으로 바꿔주는 것이 맞는가, 아니면 특정 로우의 평균을 넣어주는게 좋을지는
        # 실제로 테스트 해봐야 알 수 있다./

    @cache('offline.jbl')
    def get_offline_results(self, n_trunc, kd=50):
        """Get offline diffusion results for each gallery feature
        """
        print('[offline] starting offline diffusion')
        print('[offline] 1) prepare Laplacian and initial state')
        global trunc_ids, trunc_init, lap_alpha
        if self.use_ann:
            _, trunc_ids = self.ann.search(self.features, n_trunc)
            sims, ids = self.knn.search(self.features, kd)
            lap_alpha = self.get_laplacian(sims, ids)
        else:
            sims, ids = self.knn.search(self.features, n_trunc)
            print("sims.shape", sims.shape) #(d+q, truncate) = (5118, 1000)
            print("ids.shape", ids.shape) #(d+q, truncate) = (5118, 1000)
            trunc_ids = ids # Index value

            # 논문상에서 Laplacian 행렬 구하는 부분 La-1/2 = (I-aS)-1/2
            ## lap_alpha.shape = (5118,5118)
            lap_alpha = self.get_laplacian(sims[:, :kd], ids[:, :kd]) # 상위 50개(kd)

            print("lap_alpha..",type(lap_alpha))
            print(lap_alpha.shape)

        trunc_init = np.zeros(n_trunc)
        trunc_init[0] = 1

        print('[offline] 2) gallery-side diffusion')
        ## ci 구하기 La*ci = bi 논문에서 (9)번 식
        results = Parallel(n_jobs=-1, prefer='threads')(delayed(get_offline_result)(i)
                                      for i in tqdm(range(self.N),
                                                    desc='[offline] diffusion'))
        print(np.array(results).shape)

        ## type(result) = List, len(result) : 5118
        print("results type..",type(results))
        print("results shape..", len(results))
        all_scores = np.concatenate(results)
        print("all_scores shape..", all_scores.shape)

        print('[offline] 3) merge offline results')
        rows = np.repeat(np.arange(self.N), n_trunc)
        print("rows type..", type(rows))

        print("rows shape..", rows.shape)
        print("rows..",rows)

        # csr_matrix((data, (row, col))) : 희소행렬로 압축
        offline = sparse.csr_matrix((all_scores, (rows, trunc_ids.reshape(-1))),
                                    shape=(self.N, self.N),
                                    dtype=np.float32)
        return offline

    # @cache('laplacian.jbl')
    def get_laplacian(self, sims, ids, alpha=0.99):
        """Get Laplacian_alpha matrix
        """
        affinity = self.get_affinity(sims, ids)
        # affinity.shape = (q+g, q+g)
        num = affinity.shape[0]
        degrees = affinity @ np.ones(num) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix(
            (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
        stochastic = mat @ affinity @ mat
        sparse_eye = sparse.dia_matrix(
            (np.ones(num), [0]), shape=(num, num), dtype=np.float32)
        lap_alpha = sparse_eye - alpha * stochastic
        return lap_alpha

    # @cache('affinity.jbl')
    def get_affinity(self, sims, ids, gamma=3):
        """Create affinity matrix for the mutual kNN graph of the whole dataset
        Args:
            sims: similarities of kNN >> sims.shape : (5118, 50)
            ids: indexes of kNN >> ids.shape : (5118, 50)
        Returns:
            affinity: affinity matrix
        """

        num = sims.shape[0]
        sims[sims < 0] = 0  # similarity should be non-negative / 0보다 작은 값이 들어있으면 0으로 변경
        sims = sims ** gamma # 거듭제곱
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            ismutual[0] = False
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(sims[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                     shape=(num, num), dtype=np.float32)
        return affinity
