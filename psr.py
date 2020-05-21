import os
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from knn import KNN
from diffusion import Diffusion
from sklearn import preprocessing
from evaluate import compute_map_and_print
from scipy import sparse
np.random.seed(0)

from ast import literal_eval
import base64

org_path = "./data/psr_galleryfull.list"
db_path = './data/psr_gdatafull'

cat_dic = {
  "C10": "clothing",  "C11": "outer",  "C12": "top",  "C13": "dress",  "C14": "jumpsuit",  "C15": "suit",  "C16": "swimsuit",  "C17": "pants",  "C18": "skirts",
  "F10": "footwear",  "F11": "shoes",  "F12": "sneakers",
  "B10": "bag",  "B11": "handbag",  "B12": "backpack",  "B13": "travelbag",
  "J10": "jewelry",  "J11": "necklace",  "J12": "earring",  "J13": "bracelet",
  "A10": "accessary",  "A11": "eyewear",  "A12": "belt",  "A13": "scarves",  "A14": "gloves",  "A15": "hats",  "A16": "watch",  "A17": "necktie"
}

gal_cat_id={}
gal_cat_vec={}
for keys in cat_dic.keys():
    gal_cat_vec.update({keys:[]})
    gal_cat_id.update({keys:[]})


float32 = np.dtype('>f4')
def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=float32).tolist()


with open(org_path, "r") as st_json:
     data=st_json.readlines()
print(len(data),type(data),"\n") ;
#print(*data[:10],sep='\n'); exit()

i=0
for item in data :
    i +=1;
    dict=literal_eval(item)
    vec=decode_float_list(dict["resnet_vector"])
    cat = dict["cat_key"]
#    print(i)
#    if(cat=='MC11') :
    #result.append([cat,dict['id'],vec])
#    else : print("Not MC11", i)
#    if i==100 : break
    cat=cat[1:]
    gal_cat_vec[cat].append(vec)
    gal_cat_id[cat].append(dict['id'])

## print test
print(len(gal_cat_id),gal_cat_id.keys())
total = 0
for k in gal_cat_id.keys():
    total += len(gal_cat_id[k])
    print("\n",k,"---",len(gal_cat_id[k]),len(gal_cat_vec[k]))
    print(*gal_cat_vec[k][:3],sep='\n')

print("total num of vecs : ", total)
##########
# save vec and id
import pickle
df=open(db_path,'wb')
pickle.dump([gal_cat_id,gal_cat_vec],df)
df.close()

df=open(db_path,'rb')
gal_in= pickle.load(df)
print(type(gal_in),len(gal_in))
df.close()

gal_cat_id=gal_in[0]
gal_cat_vec=gal_in[1]
print("After : test for saving")
total = 0
for k in gal_cat_id.keys():
    total += len(gal_cat_id[k])
    print("\n",k,"---",len(gal_cat_id[k]),len(gal_cat_vec[k]))
    print(*gal_cat_vec[k][:3],sep='\n')


#print(*result[1000:1010],sep="\n")

