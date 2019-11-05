import pandas as pd
import numpy as np
import os
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
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
from surprise.model_selection import train_test_split
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

#Distribution of 836006 music-ratings
df = pd.read_csv('/Users/wuyanxu/Desktop/RS/ratings_Digital_Music.csv', names = ["user", "item", "rating", "timestep"])
data = df['rating'].value_counts().sort_index(ascending=False)
trace = go.Bar(x = data.index,
               text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],
               textposition = 'auto',
               textfont = dict(color = '#000000'),
               y = data.values,
               )
layout = dict(title = 'Distribution Of {} Music-ratings'.format(df.shape[0]),
              xaxis = dict(title = 'Rating'),
              yaxis = dict(title = 'Count'))
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

#Distribution of Number of Ratings per song
data = df.groupby('item')['rating'].count().clip(upper=50)

trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0,
                                  end = 50,
                                  size = 2))

layout = go.Layout(title = 'Distribution Of Number of Ratings Per Song',
                   xaxis = dict(title = 'Number of Ratings Per Song'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

#Distribution of Number of Ratings per user
data = df.groupby('user')['rating'].count().clip(upper=50)

trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0,
                                  end = 50,
                                  size = 2))

layout = go.Layout(title = 'Distribution Of Number of Ratings Per User',
                   xaxis = dict(title = 'Ratings Per User'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

##Find the k for KNN algorithm
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

sim_options = {'name': 'pearson',
               'user_based': True 
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNBasic(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

sim_options = {'name': 'cosine',
               'user_based': False
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNBasic(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

sim_options = {'name': 'pearson',
               'user_based': False
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNBasic(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

sim_options = {'name': 'cosine',
               'user_based': True
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNWithMeans(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

sim_options = {'name': 'pearson',
               'user_based': True
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNWithMeans(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

sim_options = {'name': 'pearson',
               'user_based': True
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNWithMeans(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

sim_options = {'name': 'cosine',
               'user_based': False
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNWithMeans(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

sim_options = {'name': 'pearson',
               'user_based': False
               }

min_mean = float("inf")
optimal_k = 1

for k in (10,20,30,40,50,60,70,80,90,100):
    algo = KNNWithMeans(sim_options=sim_options, k=k)
    x = cross_validate(algo, data, verbose=True)
    cur_mean = np.mean(x['test_rmse'])
    if(cur_mean < min_mean):
        min_mean = cur_mean
        optimal_k = k
    print("current optimal K={} min mean={}".format(optimal_k, min_mean))

data = Dataset.load_from_file(file_path, reader=reader)




benchmark = []
for algorithm in [SVD(), KNNBaseline(k=60,sim_options = {'name': 'cosine','user_based': True }), KNNBasic(), KNNWithMeans()]:
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 

benchmark = []
for algorithm in [SVD(),KNNBaseline(sim_options = {'name': 'pearson','user_based': True }), KNNBasic(k=30,sim_options = {'name': 'pearson','user_based': True }), KNNWithMeans(k=60,sim_options = {'name': 'pearson','user_based': True })]:
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')   

benchmark = []
for algorithm in [SVD(), KNNBaseline(sim_options = {'name': 'cosine','user_based': False }), KNNBasic(k=60,sim_options = {'name': 'cosine','user_based': False }), KNNWithMeans(k=90,sim_options = {'name': 'cosine','user_based': False })]:
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')   

benchmark = []
for algorithm in [SVD(), KNNBaseline(sim_options = {'name': 'pearson','user_based': False }), KNNBasic(k=80,sim_options = {'name': 'pearson','user_based': False }), KNNWithMeans(k=60,sim_options = {'name': 'pearson','user_based': False })]:
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')   

data = Dataset.load_from_file(file_path, reader=reader)
trainset, testset = train_test_split(data, test_size=.25)

algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)
result = list()
for x in testset:
    pred = algo.predict(x[0], x[1], r_ui=x[2], verbose=False)
    result.append(pred)
def custom_sort(t):
    return abs(t[3]-t[2])
result.sort(key=custom_sort)
for x in result:
    print(x)

