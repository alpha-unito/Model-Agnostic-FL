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
from openfl.utilities.data_splitters.numpy import QuantitySkewLabelsSplitter

logger = logging.getLogger(__name__)

random.seed(42)

class metabric_ShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, y, data_type, rank=1, worldsize=1, complete=False):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        splitter =  QuantitySkewLabelsSplitter(class_per_client = 2)
        event_indicator = y.iloc[:,1].values
        idx = splitter.split(event_indicator , self.worldsize)[self.rank - 1]       
        self.x = x if complete else x.iloc[idx , :]
        self.y = y if complete else  y.iloc[idx,:] 
        #self.x = x if complete else #x[self.rank - 1::self.worldsize]
        #self.y = y if complete else #y[self.rank - 1::self.worldsize]

    def get_data(self):
        return self.x, self.y

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class metabric_ShardDescriptor(ShardDescriptor):
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
        return metabric_ShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize,
            complete=complete
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['10']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['2']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'metabric dataset, shard number {self.rank}'
                    f' out of {self.worldsize}')

    def download_data(self):
        df = pd.read_csv('../metabric_dataset/metabric.csv' , sep = ",")
        df = df.iloc[:,1:]
        X = df.drop(columns = ['time','event'], axis=1)
        y = df[['time' , 'event']]
        y['event'] = y['event'].astype(bool)
        x_train , x_test , y_train , y_test = train_test_split(X,y,train_size = 0.8 ,random_state = 42)
        y_test.to_csv('../metabric_dataset/y_test.csv', index = True)
        return x_train , x_test , y_train , y_test
