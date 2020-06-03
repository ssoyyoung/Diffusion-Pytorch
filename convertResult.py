import json
import glob
import numpy as np
from multiprocessing import Pool
from elasticsearch import Elasticsearch
from collections import defaultdict

es = Elasticsearch("47.56.200.94:9200")

def queryGetES(queryID):
    index = "testset*"
    body =  {
          "query": {
            "match": {
              "_id": queryID
            }
          }
        }

    doc = es.search(
        index =index,
        body = body
    )
    
    result = doc['hits']['hits'][0]['_source']
    img_path, rb, cat =result['img_path'].split("/")[-1], result['raw_box'], result['cat_key']
    
    return img_path, rb, cat

def galleryGetESmulti(galleryIDs):
    total_body = ''
    for galleryID in galleryIDs:
        index = '{"index":"pirs*"}'
        body = '{"_source": {"includes": ["gs_bucket","id"]}, "query": {"match": {"id": "'+str(galleryID)+'"}}}'
        
        total_body = total_body+index+'\n'+body+'\n'
    
    res = es.msearch(
            body=total_body
        )
    
    gs_bucket = [res['responses'][i]['hits']['hits'][0]['_source']['gs_bucket'] for i in range(len(res['responses']))]
    
    return gs_bucket

def process(file):
    res = defaultdict(lambda: defaultdict(list))
    cate = file.split("/")[-1].split("_ranks")[0]
    
    #file list
    rank = np.load(file)[:,:10]
    score = np.load(file.replace("ranks", "scores"))
    gallery = np.load(baseDir+"pirsData/gallery/id_"+cate+".npy")
    query = np.load(baseDir+"pirsData/query/id_"+cate+".npy")
     
    for qID, ridxs, scores in zip(query.tolist(), rank, score):
        fileName, raw_box, cat = queryGetES(qID)

        res[fileName][cat].append({
                                'top_k' : galleryGetESmulti([gallery[ridx] for ridx in ridxs]),
                                'scores' : (-scores[ridxs]).tolist(),
                                'raw_box' : raw_box
                                })

    return dict(res)

def mergedicts(dict1, dict2):
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(mergedicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])

import time
s = time.time()

baseDir = "/data/github/Diffusion-Pytorch/"
fileList = glob.glob(baseDir+"pirsData/ranks/*.npy")
            
pool = Pool(processes=4)
ress = pool.map(process, fileList)

from itertools import chain

res = defaultdict(list)

for b in ress:
    res = dict(mergedicts(res, b))

with open('result.json', 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent='\t')

print(time.time() - s)