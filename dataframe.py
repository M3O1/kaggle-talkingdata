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
    DataFrame 클래스는 hdf5 포맷으로 저장된 kaggle Dataset을 읽어들이기 위해 만들어진 클래스.
    data Reference : TalkingData AdTracking Fraud Detection
    '''

    def __init__(self,
                 data_path="/home/ubuntu/data/dataset.h5",
                 key="train",
                 verbose=True):
        '''
        KGDataFrame은 필요한 데이터만을 빠르게 추출하기 위한 클래스

        구문은 Chain 형식으로 이루어져 있다.
        예시)
        1 | KGDataFrame()\  // 초기화
        2 |    .isbetween(start_datetime,end_datetime)\ // 조건문 1)
        3 |    .isin(col_name,value)\            // 조건문 2)
        4 |    .df                                // 추출
        의 구성으로 되어 있다.

        조건문을 통해, 필요한 데이터의 범위를 정하고,
        추출을 통해 데이터를 획득한다.
        이때 조건문은 중첩되어 가면서, 조건문에 모두 해당되는 row만을 추출한다.

        1) argument
            * data_path : kaggle data가 저장된 hdf5 파일 위치
            * key        : data 종류 (train or test)
            * verbose   : processing 과정을 출력할 것인지 유무
        2) method
            * select(columns) : columns에 해당하는 열만을 선택
            * isin(col_name,values) : col_name에 해당하는 열 집합 중 values에 있는 행을 선택
            * isnotin(col-name,values): col_name에 해당하는 열 집합 중 values에 없는 행을 선택
            * issame(col_name,value) : col_name에 해당하는 열 집합 중 value와 같은 행을 선택
            * isnotsame(col_name,value) : col_name에 해당하는 열 집합 중 value와 다른 행을 선택
            * isbetween(start_datetime,end_datetime) : start_time과 end_time 사이의 시간대에 있는 행을 선택
            * sample(sample_nums=None,sample_ratio=None) : 조건문에 해당하는 행 중 sample_nums 만큼 혹은 sample_ratio 비율로 랜덤하게 추출하여 선택
        '''
        if not os.path.exists(data_path):
            raise(
                "[init] file doesn't exists .... data_path : {} ".format(data_path))
        else:
            self.data_path = data_path  # the hdf5 file path containing dataset

        with pd.HDFStore(data_path) as store:
            if key in store:
                self.key = key  # key : train or test
            else:
                raise(
                    "[init]key must be in 'train' or 'test' ... current key : {}".format(key))

        self.verbose = verbose  # print if verbose is True
        self.col_names = pd.read_hdf(data_path, key, start=1, stop=2).columns
        self.cpu_counts = multiprocessing.cpu_count()

        # 조건문 집합에 해당하는 row와 column
        self.target_rows = None  # row의 index 집합
        self.target_columns = self.col_names  # column의 name 집합

    def _intersect_index(self, curr_index):
        if self.target_rows is None:
            self.target_rows = curr_index
        else:
            self.target_rows = self.target_rows.intersection(curr_index)

    def print_process(method_name):
        # 프로세스 진행상황을 보여주기 위한 decorator 함수
        def _decorator(foo):
            def magic(self, *args, **kwargs):
                start = time.time()
                if self.verbose:
                    print("[{}] QueryProcessing starts".format(method_name))
                output = foo(self, *args, **kwargs)
                end = time.time()
                if self.verbose:
                    try:
                        print("[{}] QueryProcessing ends  ---- [rows , consumed time]=[{},{:.4f}]".format(
                            method_name, self.target_rows.size, end-start))
                    except:
                        print("[{}] QueryProcessing ends  ---- [consumed time]=[{:.4f}]".format(
                            method_name, end-start))

                return output
            return magic
        return _decorator

    @print_process("select")
    def select(self, columns=[]):
        if not isinstance(columns, list):
            raise("[select] the type of column must be list")

        not_exist_column = set(columns) - set(self.col_names)
        if len(not_exist_column) > 0:
            raise(
                "[select] column[{}] doesn't exists in target dataframe-{}".format(self.col_names))

        self.target_columns = columns
        return self

    @print_process("isin")
    def isin(self, col_name=None, values=[]):
        if isinstance(values, range):
            values = list(values)
        if not isinstance(values, list):
            raise("[isin] the type of values must be list")
        if not col_name in self.col_names:
            raise("[isin] col_name must be one of the columns names...{}".format(
                self.col_names))

        with pd.HDFStore(self.data_path) as store:
            col_value = store.select_column(self.key, col_name)
        curr_index = apply_multiprocessing(
            partial(func_isin, values), col_value)
        self._intersect_index(curr_index)
        gc.collect()
        return self

    @print_process("isnotin")
    def isnotin(self, col_name=None, values=[]):
        if isinstance(values, range):
            values = list(values)
        if not isinstance(values, list):
            raise("[isnotin] the type of values must be list")
        if not col_name in self.col_names:
            raise("[isnotin] col_name must be one of the columns names...{}".format(
                self.col_names))

        with pd.HDFStore(self.data_path) as store:
            col_value = store.select_column(self.key, col_name)
        curr_index = apply_multiprocessing(
            partial(func_isnotin, values), col_value)
        self._intersect_index(curr_index)
        gc.collect()
        return self

    @print_process("issame")
    def issame(self, col_name=None, value=None):
        if not col_name in self.col_names:
            raise("col_name must be one of the columns names...{}".format(
                self.col_names))

        with pd.HDFStore(self.data_path) as store:
            col_value = store.select_column(self.key, col_name)
        curr_index = apply_multiprocessing(
            partial(func_issame, value), col_value)
        self._intersect_index(curr_index)
        gc.collect()
        return self

    @print_process("isnotsame")
    def isnotsame(self, col_name=None, value=None):
        if not col_name in self.col_names:
            raise("col_name must be one of the columns names...{}".format(
                self.col_names))

        with pd.HDFStore(self.data_path) as store:
            col_value = store.select_column(self.key, col_name)

        curr_index = apply_multiprocessing(
            partial(func_isnotsame, value), col_value)
        self._intersect_index(curr_index)
        gc.collect()
        return self

    @print_process("isbetween")
    def isbetween(self,
                  start_datetime=datetime.datetime(2017, 11, 6, 14, 0, 0),
                  end_datetime=datetime.datetime(2017, 11, 9, 16, 0, 0)):
        if not isinstance(start_datetime, datetime.datetime):
            if start_datetime is not None:
                raise("[isbetween]the type of start_datetime is not datetime")
        elif not isinstance(end_datetime, datetime.datetime):
            if end_datetime is not None:
                raise("[isbetween]the type of end_datetime is not datetime")

        with pd.HDFStore(self.data_path) as store:
            click_time = store.select_column(self.key, 'click_time')

        if start_datetime is None:
            start_datetime = click_time.iloc[0]
        if end_datetime is None:
            end_datetime = click_time.iloc[-1]

        curr_index = apply_multiprocessing(
            partial(func_isbetween, start_datetime, end_datetime), click_time)
        self._intersect_index(curr_index)
        gc.collect()
        return self

    @print_process("sample")
    def sample(self, sample_nums=None, sample_ratio=None):
        '''
        DataFrame에서 Random Sampling으로 가져옴
        sample_nums : sampling할 rows 수
        sample_ratio : sampling할 ratio
        '''
        if self.target_rows is None:
            with pd.HDFStore(self.data_path) as store:
                self.target_rows = store.select_column(self.key, "index").index
            return self

        if sample_ratio is not None:
            if sample_ratio >= 0.0 and sample_ratio <= 1.0:
                raise("[sample] sample_ratio must be in between 0.0 and 1.0")
            sample_nums = int(sample_ratio * self.target_rows.size)

        if sample_nums is not None:
            if not isinstance(sample_nums, int):
                raise("[sample] sample_nums must be integer")
            self.target_rows = self.target_rows[random.choices(range(self.target_rows.size),
                                                               k=sample_nums)].sort_values()
        gc.collect()
        return self

    @property
    def df(self):
        start = time.time()
        if self.verbose:
            print("start to get dataframe from File")
        
        if self.target_rows is None:
            output = pd.read_hdf(self.data_path, self.key+"_raw").loc[:,self.target_columns]
        else:
            if len(self.target_rows)>=10**7:
                output = pd.read_hdf(self.data_path, self.key+"_raw").loc[self.target_rows,self.target_columns]
            else:
                output = pd.read_hdf(self.data_path, self.key, where=self.target_rows, columns=self.target_columns)
        gc.collect()

        if self.verbose:
            print("consumed time ---- {}".format(time.time()-start))
        return output

    def get_dataframe(self):
        gc.collect()
        return self.df

###########################################
# Query Processing을 병렬 처리 하기 위한 helper 메소드들
#
#   apply_multiprocessing : 병렬 처리를 위한 map 함수
#   func_* : 병렬 처리를 위한 reduce 함수
###########################################


def apply_multiprocessing(func, series):
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)

    result = pool.map_async(func, np.array_split(series, cpus))
    index_list = sum(result.get(), [])

    pool.close()
    return pd.Index(index_list)


def func_isin(values, chunk):
    return list(chunk[chunk.isin(values)].index)


def func_isnotin(values, chunk):
    return list(chunk[~chunk.isin(values)].index)


def func_issame(value, chunk):
    return list(chunk[chunk == value].index)


def func_isnotsame(value, chunk):
    return list(chunk[chunk != value].index)


def func_isbetween(start, end, chunk):
    return list(chunk[(chunk >= start) & (chunk <= end)].index)

###########################################
# csv 파일로부터 hdf5파일을 만드는 함수
###########################################


def set_hdf5_from_csv(csv_path="/home/ubuntu/data/train.csv",
                      hdf5_path="/home/ubuntu/data/dataset.h5",
                      key='train'):
    if key == 'train':
        train_df = pd.read_csv(csv_path,
                               parse_dates=["click_time", "attributed_time"])
        # difference time for attributed time and click time
        train_df['diff_time'] = train_df.attributed_time - train_df.click_time
        # set unique_id
        train_df['unique_id'] = train_df.ip + \
            train_df.os * (10**6) + train_df.device * (10**9)
        # convert to boolean type
        train_df.is_attributed = train_df.is_attributed.astype(bool)
        # Save to hdf5 format
        train_df.to_hdf(hdf5_path, key=key, format="table", data_columns=True)
        train_df.to_hdf(hdf5_path, key=key+"_raw",format='fixed')
    elif key == 'test':
        # difference time for attributed time and click time
        test_df = pd.read_csv(csv_path, parse_dates=['click_time'])
        # set unique_id
        test_df['unique_id'] = test_df.ip + test_df.os * \
            (10**6) + test_df.device * (10**9)
        # Save to hdf5 format
        test_df.to_hdf(hdf5_path, key=key, format='table', data_columns=True)
        test_df.to_hdf(hdf5_path, key=key+"_raw",format='fixed')
