import base64
import numpy as np
from ast import literal_eval
import pickle
import os.path

# TODO : npy 파일 생성
# TODO : 카테고리 별 OFFLINE 파일 생성
# TODO : 데이터 정규화

# decode vector
float32 = np.dtype('>f4')

# define path
json_path = "full_dataset.json"
pkl_path = "./pirsData/pirsData.pkl"

def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=float32).tolist()

def create_pkl():
    total_vec, total_id = {}, {}

    with open(json_path, "r") as f: origindata = f.readlines()
    for data in origindata:
        data = literal_eval(data)
        vec = np.asarray(decode_float_list(data['resnet_vector']))
        cate = data["cat_key"][1:]
        id = data['id']

        if not cate in total_vec.keys(): total_vec[cate], total_id[cate] = [], []

        total_vec[cate].append(vec)
        total_id[cate].append(id)

    with open(pkl_path, "wb") as file:
        pickle.dump({"total_vec": total_vec, "total_id": total_id}, file)
    
        

def read_file():
    # Loading pickle data
    if not os.path.isfile(pkl_path): create_pkl()
    with open(pkls_path, "rb") as f: data = pickle.load(f)
    print("total_vec..", len(data['total_vec']), "total_id..", len(data['total_id']))

    total_vec = data['total_vec']['A12']
    total_id = data['total_id']['A12']

    vector = np.array(total_vec) #(doc_count, vec_length)
    id = np.array(total_id)
    print("vector.shape..", vector.shape)
    print("id.shape..", id.shape)


if __name__ == "__main__":
    read_file()



'''
PIRS data ... 3190868
vec : A12 ..... 44854
vec : A14 ..... 60751
vec : A15 ..... 91983
vec : A16 ..... 106773
vec : A17 ..... 13200
vec : B11 ..... 120389
vec : B12 ..... 52855
vec : C11 ..... 450916
vec : C12 ..... 390370
vec : C15 ..... 49532
vec : C16 ..... 36101
vec : C17 ..... 266563
vec : F11 ..... 506285
vec : F12 ..... 398316
vec : A11 ..... 75882
vec : A13 ..... 13731
vec : C13 ..... 67828
vec : C18 ..... 13858
vec : B13 ..... 24260
vec : J11 ..... 157362
vec : J12 ..... 131963
vec : J13 ..... 117096
'''


category= ['C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
            'F10', 'F11', 'F12', 'B10', 'B11', 'B12', 'B13',
            'J10', 'J11', 'J12', 'J13',s
            'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17']

cate_dic = {
  "C10": "clothing",  "C11": "outer",  "C12": "top",  "C13": "dress",  "C14": "jumpsuit",  "C15": "suit",  "C16": "swimsuit",  "C17": "pants",  "C18": "skirts",
  "F10": "footwear",  "F11": "shoes",  "F12": "sneakers",
  "B10": "bag",  "B11": "handbag",  "B12": "backpack",  "B13": "travelbag",
  "J10": "jewelry",  "J11": "necklace",  "J12": "earring",  "J13": "bracelet",
  "A10": "accessary",  "A11": "eyewear",  "A12": "belt",  "A13": "scarves",  "A14": "gloves",  "A15": "hats",  "A16": "watch",  "A17": "necktie"
}

