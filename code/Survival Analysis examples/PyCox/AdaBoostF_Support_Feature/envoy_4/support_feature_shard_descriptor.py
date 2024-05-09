# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import pandas as pd
import numpy as np 
import logging
from typing import List
import random

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from sklearn.model_selection import train_test_split

from sklearn.compose import make_column_selector
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import OneHotEncoderDF, ColumnTransformerDF, SimpleImputerDF, StandardScalerDF
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from openfl.utilities.data_splitters.numpy import QuantitySkewLabelsSplitter
logger = logging.getLogger(__name__)

random.seed(42)

class support_feature_ShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1, complete=False):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize   
        self.x = x if complete else x[self.rank - 1::self.worldsize]
        self.y = y if complete else y[self.rank - 1::self.worldsize]
           

    def get_data(self):
        return self.x, self.y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class support_feature_ShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        x_train, x_test, y_train, y_test = self.download_data()

        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train', complete=False):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return support_feature_ShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize,
            complete=complete
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['55']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['2']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'support_feature dataset, shard number {self.rank}'
                    f' out of {self.worldsize}')
    def _get_preprocess_transformer(self) -> ColumnTransformerDF:
        sel_fac = make_column_selector(pattern='^fac\\_')
        enc_fac = PipelineDF(steps=[('ohe', OneHotEncoderDF(sparse=False, drop='if_binary', handle_unknown='ignore'))])
        sel_num = make_column_selector(pattern='^num\\_')
        enc_num = PipelineDF(steps=[('impute', SimpleImputerDF(strategy='median')), ('scale', StandardScalerDF())])
        tr = ColumnTransformerDF(transformers=[('ohe', enc_fac, sel_fac), ('s', enc_num, sel_num)])
        return tr
    def _split_dataframe(self , df: pd.DataFrame, split_size=0.2) :
        return train_test_split(df, stratify=df['event'], test_size=split_size)
    def download_data(self):
        df_train = pd.read_csv('../support_feature_dataset/support_preprocessed.csv' , sep = ",")
        df_train = df_train.iloc[:,1:]
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        cols_leave =  ['sex', 'income', 'diabetes', 'dementia', 'adlp', 'adls', 'dzgroup_ARF/MOSF w/Sepsis', 'dzgroup_CHF', 'dzgroup_COPD', 'dzgroup_Cirrhosis', 'dzgroup_Colon Cancer', 'dzgroup_Coma', 'dzgroup_Lung Cancer', 'dzgroup_MOSF w/Malig', 'dzclass_ARF/MOSF', 'dzclass_COPD/CHF/Cirrhosis', 'dzclass_Cancer', 'dzclass_Coma', 'race_asian', 'race_black', 'race_hispanic', 'race_other', 'race_white', 'ca_metastatic', 'ca_no', 'ca_yes', 'sfdm2_2 mo. follow-up', 'sfdm2_Coma or Intub', 'sfdm2_SIP30', 'sfdm2_adl4 (5 if sur)', 'sfdm2_no(M2 and SIP pres)']
        cols_standardize =['age', 'slos', 'num.co', 'edu', 'scoma', 'charges', 'totcst', 'avtisst', 'hday', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlsc']
   

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]

        x_mapper = DataFrameMapper(standardize + leave)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        get_target = lambda df: np.array([df['time'].values, df['event'].values])
        y_train = get_target(df_train)
        y_test = get_target(df_test)
        y_train = pd.DataFrame(data=y_train.T, columns=['time', 'event'])
        y_test = pd.DataFrame(data=y_test.T, columns=['time', 'event'])
        return x_train , x_test , y_train , y_test












