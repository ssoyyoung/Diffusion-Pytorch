#!/usr/bin/env python
# -*- coding: utf-8 -*-

" rank module "

import os
import sys
import time
import pickle
import argparse
import numpy as np
from dataset import Dataset
from diffusion import Diffusion
from sklearn import preprocessing


# tm > box : query image
# 결과 확인 : elastic_get_path
# main : rank.py

def search():
    st = time.time()
    n_query = len(queries)
    print("min..", np.min(queries))
    print("max..", np.max(queries))
    diffusion = Diffusion(np.vstack([queries, gallery]), args.cache_dir)
    print("diffusion shape",diffusion.features.shape)

    # offline type : scipy.sparse.csr.csr_matrix (희소행렬)
    # offline shape : (5118 ,5118) >> query와 gallery의 합
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)
    # offline.data 안에 nan 값이 존재함 
    print("offline..", len(offline.data))

    print("offline features shape..",offline.shape)

    offline.data = np.nan_to_num(offline.data, copy=False)

    # 컬럼별로 확률화(SUM=1), feature=(1-a)(1-aS)의 역행렬
    features = preprocessing.normalize(offline, norm="l2", axis=1)
    print("features..",features.shape)

    scores = features[:n_query] @ features[n_query:].T
    np.save("pirsData/scores/"+ args.cate +"_scores.npy", scores)

    print("1> features[:n_query].shape :", features[:n_query].shape)
    print("2> features[n_query:].shape :", features[n_query:].shape)
    print("3> scores.shape :", scores.shape)
    # scores.shape : (55, 5063) = (쿼리, 갤러리) = (row, col)

    ranks = np.argsort(-scores.todense())
    np.save("pirsData/ranks/"+ args.cate +"_ranks.npy", ranks)
    print("ranks[0]...\n", ranks[:10,:10])
    print("time check...>>>", args.cate,">>>", time.time()-st, ">>>", len(queries))
    # np.argsort : 행렬 안에서 값이 작은 값부터 순서대로 데이터의 INDEX 반환



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',
                        type=str,
                        default='./cache',
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help="""
                        Name of the dataset
                        """)
    parser.add_argument('--query_path',
                        type=str,
                        required=True,
                        help="""
                        Path to query features
                        """)
    parser.add_argument('--gallery_path',
                        type=str,
                        required=True,
                        help="""
                        Path to gallery features
                        """)
    parser.add_argument('--gnd_path',
                        type=str,
                        help="""
                        Path to ground-truth
                        """)
    parser.add_argument('-n', '--truncation_size',
                        type=int,
                        default=1000,
                        help="""
                        Number of images in the truncated gallery
                        """)
    parser.add_argument('--cate',
                        type=str,
                        help="""
                        PIRS-category
                        """)

    args = parser.parse_args()
    args.kq, args.kd = 10, 50

    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.cache_dir): os.makedirs(args.cache_dir)
    if not os.path.isfile(args.query_path): 
        print(args.cate, "not query file")
        sys.exit()

    # IN PAPER : DATABASE = X, QUERY = Q
    dataset = Dataset(args.query_path, args.gallery_path)
    queries, gallery = dataset.queries, dataset.gallery

    if len(gallery) > 100000:
        args.truncation_size = 2000
        args.kd = 200

    # queries = np.array([[1, 0]], dtype=np.float32)
    # gallery = np.array([[0, 1], [1, 2], [1, 3], [2, -1]], dtype=np.float32)
    # queries : float32 / gallery = float32

    print("\n<< ", args.cate, " >>")
    print("#######################################")
    print("INFO",
          "\nquery & gallery path", args.query_path, args.gallery_path,
          "\nquery..", queries.shape,
          "\ngallery..", gallery.shape,
          "\ncount, dimension..", np.vstack([queries, gallery]).shape,
          "\ntruncation_size, kd..", args.truncation_size, args.kd)

    print("#######################################")

    

    if os.path.isfile("pirsData/ranks/"+ args.cate +"_ranks.npy"):
        print("already get offline and rank result!")
    elif len(gallery) > 100000:
        print("ann search! >> TODO")
    else:
        search()

