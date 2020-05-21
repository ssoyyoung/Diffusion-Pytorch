#!/usr/bin/env python
# -*- coding: utf-8 -*-

" rank module "

import os
import argparse
import numpy as np
from dataset import Dataset
from diffusion import Diffusion
from sklearn import preprocessing


# tm > box : query image
# 결과 확인 : elastic_get_path
# main : rank.py

def search():
    n_query = len(queries)
    diffusion = Diffusion(np.float32(np.vstack([queries, gallery])), args.cache_dir)

    # offline type : scipy.sparse.csr.csr_matrix (희소행렬)
    # offline shape : (5118 ,5118) >> query와 gallery의 합
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)
    print("offline..", len(offline.data))

    # 컬럼별로 확률화(SUM=1), feature=(1-a)(1-aS)의 역행렬
    features = preprocessing.normalize(offline, norm="l2", axis=1)
    print("features..",features.shape)

    scores = features[:n_query] @ features[n_query:].T
    print("1", features[:n_query].shape)
    print("2", features[n_query:].shape)
    print("3", (features[n_query:].T).shape)
    print("4", scores.shape)
    print("type(scores)",type(scores))
    # scores.shape : (55, 5063) = (쿼리, 갤러리) = (row, col)
    print(type(-scores.todense()))

    ranks = np.argsort(-scores.todense())
    print("ranks[0] query result...", ranks[0])
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
    args = parser.parse_args()
    args.kq, args.kd = 10, 50
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.cache_dir): os.makedirs(args.cache_dir)

    dataset = Dataset(args.query_path, args.gallery_path)
    queries, gallery = dataset.queries, dataset.gallery

    print("\n#######################################")
    print("INFO",
          "\nquery..", queries.shape,
          "\ngallery..", gallery.shape,
          "\ncount, dimension..", np.vstack([queries, gallery]).shape)
    print("#######################################")

    search()
