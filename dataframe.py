import os
import datetime
import time
import random
import types

import gc
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial

class KGDataFrame(object):
    '''
    DataFrame 클래스는 hdf5 포맷으로 저장된 Dataset을 읽어들이기 위해 만들어진 클래스.
    '''
    def __init__(self,data_path="/home/ubuntu/data/dataset.h5",key="train",verbose=True):
        # data_path 체크
        if not os.path.exists(data_path):
            raise("[__init__] Data file doesn't exists .... data_path : {} ".format(data_path))
        else:
            self.data_path = data_path # the hdf5 file path containing dataset 
        
        with pd.HDFStore(data_path) as store:
            if key in store:
                self.key = key # key : train or test
            else:
                raise("[__init__]key must be in 'train' or 'test' ... current key : {}".format(key))
        self.verbose = verbose # print if verbose is True
        self.col_names = pd.read_hdf(data_path,key,start=1,stop=2).columns
        self.cpu_counts = multiprocessing.cpu_count()
        # Constraints
        self.target_index = None
        self.target_columns = None

    def _intersect_index(self,curr_index):
        if self.target_index is None:
            self.target_index = curr_index
        else:
            self.target_index = self.target_index.intersection(curr_index)    
    
    def print_process(method_name):
        def _decorator(foo):
            def magic(self, *args, **kwargs):
                start = time.time()
                if self.verbose:
                    print("[{}] QueryProcessing starts".format(method_name))
                output =  foo(self, *args, **kwargs)
                end = time.time()
                if self.verbose:
                    try:
                        print("[{}] QueryProcessing ends  ---- [rows , consumed time]=[{},{:.4f}]".format(
                    method_name,self.target_index.size,end-start))
                    except:
                        print("[{}] QueryProcessing ends  ---- [consumed time]=[{:.4f}]".format(
                    method_name,end-start))
                        
                return output
            return magic
        return _decorator
    
    @print_process("select")
    def select(self,columns=[]):
        if not isinstance(columns,list):
            raise("[select] the type of column must be list")
        
        not_exist_column = set(columns) - set(self.col_names)
        if len(not_exist_column) > 0 :
            raise("[select] column[{}] doesn't exists in target dataframe-{}".format(self.col_names))
        
        self.target_columns = columns
        return self
    
    @print_process("isin")
    def isin(self,col_name=None,values=[]):
        if isinstance(values,range):
            values = list(values)
        if not isinstance(values,list):
            raise("[isin] the type of values must be list")
        if not col_name in self.col_names:
            raise("[isin] col_name must be one of the columns names...{}".format(self.col_names))
            
        with pd.HDFStore(self.data_path) as store:
            col_value = store.select_column(self.key,col_name)
        curr_index = apply_multiprocessing(partial(func_isin,values),col_value)
        self._intersect_index(curr_index)
        gc.collect()
        return self
    
    @print_process("issame")
    def issame(self,col_name=None,value=None):
        if not col_name in self.col_names:
            raise("col_name must be one of the columns names...{}".format(self.col_names))

        with pd.HDFStore(self.data_path) as store:
            col_value = store.select_column(self.key,col_name)
        curr_index = apply_multiprocessing(partial(func_issame,value),col_value)
        self._intersect_index(curr_index)
        gc.collect()
        return self
    
    @print_process("isnotsame")
    def isnotsame(self,col_name=None,value=None):
        if not col_name in self.col_names:
            raise("col_name must be one of the columns names...{}".format(self.col_names))

        with pd.HDFStore(self.data_path) as store:
            col_value = store.select_column(self.key,col_name)
        
        curr_index = apply_multiprocessing(partial(func_isnotsame,value),col_value)
        self._intersect_index(curr_index)
        gc.collect()
        return self
    
    @print_process("isbetween")
    def isbetween(self,start_time=datetime.datetime(2017,11,6,14,0,0),
                  end_time=datetime.datetime(2017,11,9,16,0,0)):
        if not isinstance(start_time,datetime.datetime):
            if start_time is not None:        
                raise("[isbetween]the type of start_time is not datetime")
        elif not isinstance(end_time,datetime.datetime):
            if end_time is not None:
                raise("[isbetween]the type of end_time is not datetime")
        
        with pd.HDFStore(self.data_path) as store:
            click_time = store.select_column(self.key,'click_time')
            
        if start_time is None:
            start_time = click_time.iloc[0]
        if end_time is None:
            end_time = click_time.iloc[-1]
        
        curr_index = apply_multiprocessing(partial(func_isbetween,start_time,end_time),click_time)
        self._intersect_index(curr_index)
        gc.collect()
        return self
    
    @print_process("sample")
    def sample(self,sample_nums=None,sample_ratio=None):
        '''
        DataFrame에서 Random Sampling으로 가져옴
        sample_nums : sampling할 rows 수
        sample_ratio : sampling할 ratio
        '''
        if self.target_index is None:
            with pd.HDFStore(self.data_path) as store:
                self.target_index = store.select_column(self.key,"index").index
            return self
        
        if sample_ratio is not None:
            if sample_ratio >= 0.0 and sample_ratio <= 1.0:
                raise("[sample] sample_ratio must be in between 0.0 and 1.0")
            sample_nums = int(sample_ratio * self.target_index.size)
        
        if sample_nums is not None:
            if not isinstance(sample_nums,int): 
                raise("[sample] sample_nums must be integer")
            self.target_index = self.target_index[random.choices(range(self.target_index.size),
                                                                 k=sample_nums)].sort_values()
        gc.collect()
        return self
    
    @property
    def df(self):
        start = time.time()
        if self.verbose: print("start to get dataframe from File")
        if self.target_index is None:
            if self.verbose: print("consumed time ---- {}".format(time.time()-start))
            return pd.read_hdf(self.data_path,self.key,columns=self.target_columns)
        else:
            if self.verbose: print("consumed time ---- {}".format(time.time()-start))
            return pd.read_hdf(self.data_path,self.key,where=self.target_index,columns=self.target_columns)
        
    def get_dataframe(self):
        gc.collect()
        return self.df
    
def apply_multiprocessing(func, series):
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)

    result = pool.map_async(func,np.array_split(series,cpus))
    index_list = sum(result.get(),[])

    pool.close()
    return pd.Index(index_list)

def func_isin(values,chunk):
    return list(chunk[chunk.isin(values)].index)

def func_issame(value,chunk):
    return list(chunk[chunk==value].index)

def func_isnotsame(value,chunk):
    return list(chunk[chunk!=value].index)

def func_isbetween(start,end,chunk):
    return list(chunk[(chunk >= start) & (chunk <= end)].index)