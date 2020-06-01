import os
import glob
import base64
import numpy as np
from ast import literal_eval


# decode vector
#float32 = np.dtype('f4')

def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=np.float32).tolist()


def create(fileName):
    with open(fileName, "r") as f: originData = f.readlines()

    total_vec = []
    total_id = []

    for data in originData:
        data = literal_eval(data)
        vec = np.asarray(decode_float_list(data['resnet_vector']))
        _id = np.asarray(data['id'])

        total_vec.append(vec)
        total_id.append(_id)
        
        
    saveName = "pirsData/gallery/"+fileName.split("/")[-1].split(".")[0]
    np.save(saveName.replace("resnet1024_*","vector")+".npy", np.array(total_vec))
    np.save(saveName.replace("resnet1024_*","id")+".npy", np.array(total_id))
    print(">>>", saveName, "saved!")


def queryCreate(fileName):
    with open(fileName, "r") as f: originData = f.readlines()

    total_vec = {}
    total_id = {}

    for data in originData:
        data = literal_eval(data)
        vec = np.asarray(decode_float_list(data['resnet_vector']))
        _id = np.asarray(data['id'])
        cate = data["cat_key"]

        if not cate in total_vec.keys(): total_vec[cate], total_id[cate] = [], []

        total_vec[cate].append(vec)
        total_id[cate].append(_id)

    for key in total_vec.keys():
        k = key.lower()[0]+"_"+key[1:]+".npy"
        np.save("pirsData/query/vector_"+k, np.array(total_vec[key]))
        np.save("pirsData/query/id_"+k, np.array(total_id[key]))
        print("query :", k, "saved!")


if __name__ == "__main__":

    

    # create gallery npy
    fileList = glob.glob("/data/github/elastic/*.*")

    for fileN in fileList:
        saveCheck="pirsData/gallery/"+(fileN.split("/")[-1].split(".")[0]).replace("resnet1024_*","vector")+".npy"
        if os.path.isfile(saveCheck): 
            print(saveCheck, "already saved..!")
            continue
        create(fileN)

    