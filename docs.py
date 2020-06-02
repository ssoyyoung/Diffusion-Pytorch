import glob
import numpy as np
from elasticsearch import Elasticsearch

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

def galleryGetES(galleryID):
    index = "pirs*"
    body =  {
          "query": {
            "match": {
              "id": int(galleryID)
            }
          }
        }

    doc = es.search(
        index =index,
        body = body
    )
   
    return doc['hits']['hits'][0]['_source']['gs_bucket']



baseDir = "/data/github/Diffusion-Pytorch/"
fileList = glob.glob(baseDir+"pirsData/ranks/*.npy")

result = {}
for file in fileList:
    cate = file.split("/")[-1].split("_ranks")[0]
    
    #file list
    rank = np.load(file)[:,:10]
    score = np.load(file.replace("ranks", "scores"))[:,:10]
    gallery = np.load(baseDir+"pirsData/gallery/id_"+cate+".npy")
    query = np.load(baseDir+"pirsData/query/id_"+cate+".npy")
       
    
    for q in range(len(query)):
        img_path, rb, cat = queryGetES(q)
        
        
        
        
    result = {}
    
    #query
    queryBody = {

    }
    
    
    #gallery
    for k in range(10):
        id = gallery[rank[0]]
          
                                 
    rank = rank[:, :10] # top K
    scores = scores[:, :10]
                          
    
    