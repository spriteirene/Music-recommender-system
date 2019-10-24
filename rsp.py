import pandas as pd
import numpy as np
import os
from plotly.offline import init_notebook_mode, plot
from surprise import SVD
from surprise import Dataset
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise.reader import Reader 
from surprise.model_selection import cross_validate
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
import random

## Explorary the data
data = pd.read_csv('/Users/wuyanxu/Desktop/RS/ratings_Digital_Music.csv', names = ["user", "item", "rating", "timestep"])
x = data.groupby(["user"]).count()
data_ = x.loc[x["item"] >= 5]
#print(data_) there are 22,878 unique users have rated more than five items
merge_table = pd.merge(data_, data, on = "user", how = "inner")
#print(merge_table)
merge_table= merge_table.drop(columns = ["item_x", "rating_x", "timestep_x"])
merge_table = merge_table.rename(columns = {"item_y" : "item", "rating_y" : "rating", "timestep_y": "timestep"})
m = merge_table.groupby(["item"]).count()
m = m.loc[m["user"] >= 5 ]
merge_table = pd.merge(m, merge_table, on = "item", how = "inner")
data_final = merge_table.drop(columns = ["user_x", "rating_x", "timestep_x", "timestep_y"])
data_final = data_final.rename(columns = {"user_y": "users", "rating_y": "ratings"})
#print(data_final)　
len(data_final["item"].unique()) #6,909 unique items
len(data_final["users"].unique()) #17,659 unique users
export_csv = data_final.to_csv ('/Users/wuyanxu/Desktop/finaldata.csv', index = None, header=None)

random.seed(1024)
file_path = os.path.expanduser('/Users/wuyanxu/Desktop/finaldata.csv')

reader = Reader(line_format='item user rating', sep=',')

data = Dataset.load_from_file(file_path, reader=reader)

sim_options = {'name': 'cosine',
               'user_based': True 
               }

min_mean = float("inf")
optimal_k = 1

for k in [10,20,30,40,50,60,70,80,90,100]:
    algo = KNNBasic(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))