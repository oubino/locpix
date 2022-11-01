"""Datastruc module.

This module contains definitions of the datastructures the
SMLM dataitem will be parsed as during processing.
"""

import os
import torch
from torch_geometric.data import Dataset, HeteroData, Data
import pyarrow.parquet as pq
import pyarrow.compute as pc
import ast
import polars as pl


class SMLMDataset(Dataset):
    """SMLM dataset class.

    All the SMLM dataitems are defined in this class.
    Assumption that name is the last part of the file 
    name before the .file_extension.

    Attributes:
        heterogeneous (bool): If True then separate graph per channel
            i.e. heterogeneous data.
            If False then one graph for all data i.e. homogeneous 
            graph
        raw_dir_root: A string with the directory of the the folder
            which contains the "raw" dataset i.e. the parquet files,
            is not technically raw as has passed through
            our preprocessing module - bear this in mind
        processed_dir_root: A string with the directory of the the folder
            which contains the the directory of the
            processed dataset - processed via pygeometric
            i.e. Raw .csv -> Preprocessing module outputs to
            raw_dir -> Taken in to data_loading module processed
            to processed_dir -> Then pytorch analysis begins
        transform: The transform to be applied to each
                   loaded in graph/point cloud.
        data_list: A list of the data from the dataset
            so can access via a numerical index later.
    """

    def __init__(self, heterogeneous, raw_dir_root, processed_dir_root,
                 transform=None, pre_transform=None, pre_filter=None):
        """Inits SMLMDataset with root directory where
        data is located and the transform to be applied when
        getting item.
        Note the pre_filter (non callable) is boolean? whether 
        there is a pre-filter
        Note the pre_filter (callable) takes in data item and returns
        whether it should be included in the final dataset"""

        self.heterogeneous = heterogeneous
        # index the dataitems (idx)
        self._raw_dir_root = raw_dir_root
        self._processed_dir_root = processed_dir_root
        self._raw_file_names = list(sorted(os.listdir(raw_dir_root)))
        self._processed_file_names = list(sorted(os.listdir(processed_dir_root)))
        # Note deliberately set root to None
        # as going to overload the raw and processed
        # dir. This could cause problems so be aware
        super().__init__(None, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir_root

    @property
    def processed_dir(self) -> str:
        return self._processed_dir_root

    @property
    def raw_file_names(self):
        return self._raw_file_names
    
    @property 
    def processed_file_names(self):
        return self._processed_file_names

    def process(self):
        if self.heterogeneous:
            self.process_heterogeneous()
        elif self.heterogeneous is False:
            self.process_homogeneous()

    def process_heterogeneous(self):
        """Process the raw data into procesed data.
        This currently includes
            1. For each .parquet create a heterogeneous graph
            , where the different (i.e. heterogeneous) nodes
            are due to there being multiple channels.
            e.g. two channel image with 700 localisations for 
            channel 0 and 300 for channel 1 - would have 
            1000 nodes and each node is type (channel 0 or 
            channel 1)
            2. Then if not pre-filtered the heterogeneous 
            graph is pre-transformed
            3. Then the graph is saved"""

        idx = 0
        idx_to_name = {}

        # convert raw parquet files to tensors 
        for raw_path in self.raw_paths:
            # read in parquet file
            arrow_table = pq.read_table(raw_path)
            # dimensions and channels metadata
            dimensions = arrow_table.schema.metadata[b'dim']
            channels = arrow_table.schema.metadata[b'channels']
            dimensions = int(dimensions)
            channels = ast.literal_eval(channels.decode("utf-8"))
            # each dataitem is a heterogeneous graph
            # where the channels define the different type of node
            # i.e. for two channel data have two types of node
            # for both channels
            data = HeteroData()
            # for channel in list of channels
            for chan in channels:
                # filter table
                filter = pc.field("channel") == chan
                filter_table = arrow_table.filter(filter)
                # convert to tensor (Number of points x 2/3 (dimensions))
                x = torch.from_numpy(filter_table['x'].to_numpy())
                y = torch.from_numpy(filter_table['y'].to_numpy())
                if dimensions == 2:
                    coord_data = torch.stack((x, y), dim=1)
                if dimensions == 3:
                    z = torch.from_numpy(arrow_table['z'].to_numpy())
                    coord_data = torch.stack((x, y, z), dim=1)

                # feature tensor
                # shape: [Number of points x 2/3 dimensions]
                data[str(chan)].x = coord_data

                # position tensor
                # shape: [Number of points x 2/3 dimensions] 
                data[str(chan)].pos = coord_data

                # localisation level labels
                data[str(chan)].y = torch.from_numpy(filter_table['gt_label']
                                                     .to_numpy())
                
            # if pre filter skips it then skip this item
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            # pre-transform
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # save it
            _, extension = os.path.splitext(raw_path)
            _, tail = os.path.split(raw_path)
            file_name = tail.strip(extension)
            torch.save(data, os.path.join(self.processed_dir,
                                          f'{idx}.pt'))
            
            # add to index
            idx_to_name['idx'].append(idx)
            idx_to_name['file_name'].append(file_name)
            idx += 1

        # save mapping from idx to name
        df = pl.from_dict(idx_to_name)
        df.write_csv(os.path.join(self.processed_dir, 'file_map.csv'))
        
    def process_homogeneous(self):
        """Process the raw data into procesed data.
        This currently includes
            1. For each .parquet create a homogeneous graph
            2. Then if not pre-filtered the 
            graph is pre-transformed
            3. Then the graph is saved"""
        
        idx = 0
        idx_to_name = {'idx':[],'file_name':[]}

        # convert raw parquet files to tensors 
        for raw_path in self.raw_paths:
            # read in parquet file
            arrow_table = pq.read_table(raw_path)
            # dimensions metadata
            dimensions = arrow_table.schema.metadata[b'dim']
            dimensions = int(dimensions)
            # each dataitem is a homogeneous graph
            data = Data()
            
            # convert to tensor (Number of points x 2/3 (dimensions))
            x = torch.from_numpy(arrow_table['x'].to_numpy())
            y = torch.from_numpy(arrow_table['y'].to_numpy())
            if dimensions == 2:
                coord_data = torch.stack((x, y), dim=1)
            if dimensions == 3:
                z = torch.from_numpy(arrow_table['z'].to_numpy())
                coord_data = torch.stack((x, y, z), dim=1)
            
            # feature tensor
            # shape: [Number of points x 2/3 dimensions]
            data.x = coord_data

            # position tensor
            # shape: [Number of points x 2/3 dimensions] 
            data.pos = coord_data

            # localisation level labels
            data.y = torch.from_numpy(arrow_table['gt_label']
                                                    .to_numpy())
            
            # if pre filter skips it then skip this item
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            # pre-transform
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # save it
            _, extension = os.path.splitext(raw_path)
            _, tail = os.path.split(raw_path)
            file_name = tail.strip(extension)
            torch.save(data, os.path.join(self.processed_dir,
                                          f'{idx}.pt'))

            # add to index
            idx_to_name['idx'].append(idx)
            idx_to_name['file_name'].append(file_name)
            idx += 1

        # save mapping from idx to name
        df = pl.from_dict(idx_to_name)
        df.write_csv(os.path.join(self.processed_dir, 'file_map.csv'))


    def len(self):
        files = self._processed_file_names
        if 'pre_filter.pt' in files:
            files.remove('pre_filter.pt')
        if 'pre_transform.pt' in files:
            files.remove('pre_transform.pt')
        if 'file_map.csv' in files:
            files.remove('file_map.csv')
        return len(files)

    def get(self, idx):
        """I believe that pytorch geometric is wrapper 
        over get item and therefore it handles the 
        transform"""
        data = torch.load(os.path.join(self.processed_dir, f'{idx}.pt'))
        return data
