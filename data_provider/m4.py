# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm
import logging
import os
import pathlib
import sys
from urllib import request

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pandas import read_csv
from datetime import datetime


def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''


def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\rDownloading {} to {} {:.1f}%'.format(url, file_path, progress_pct))
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')


@dataclass()
class M4Dataset:
    #ids: np.ndarray
    #groups: np.ndarray
    #frequencies: np.ndarray
    #horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training, dataset_file, N_REGRESS, MAX_STEPS) -> 'M4Dataset':
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        info_file = os.path.join(dataset_file, 'M4-info.csv')
        #train_cache_file = os.path.join(dataset_file, 'training.npz')
        #test_cache_file = os.path.join(dataset_file, 'test.npz')
        #m4_info = pd.read_csv(info_file)
        
        def parser(x):
                return datetime.strptime(x, "%Y-%m-%dT%H:%M+10:00")
		
        series_or = read_csv('./dataset/m4/aemo_2018.csv', header=0, parse_dates=[0], sep=",", index_col=0, date_parser=parser)
        grupo_usi=['GUNNING1','GULLRWF1','CAPTL_WF','TARALGA1','CULLRGWF','WOODLWN1']
        t_ini_tr=datetime.strptime("02/01/2018 00:00", "%d/%m/%Y %H:%M")
        t_fim_tr=datetime.strptime("30/08/2018 00:00", "%d/%m/%Y %H:%M")
        t_ini_ts=datetime.strptime("01/09/2018 00:00", "%d/%m/%Y %H:%M")
        t_fim_ts=datetime.strptime("30/12/2018 00:00", "%d/%m/%Y %H:%M")
        #N_REGRESS = self.args.seq_len   #12 nº de regressores
        #MAX_STEPS = self.args.pred_len    # horizonte máximo de previsão na simulação

        usi=grupo_usi[0]		
        series_ds=pd.Series(series_or.loc[:,usi])
        series_ds = series_ds.resample('30T').mean()  # converte para 30min
        series_ds=pd.Series(series_ds.values, index=series_ds.index.shift(1, freq='25T'))
        pot = series_ds.max()
        series_ds=series_ds/pot
        series_ds[series_ds<0]=0
        series_ds[series_ds.isnull()]=0
        #
        series_temp=series_ds
        for i in range(1,N_REGRESS + MAX_STEPS):
            temp=series_ds.shift(i)
            series_temp=pd.concat([temp,series_temp],axis=1)
        #
        series_temp1=series_temp[t_ini_tr:t_fim_tr]
        series_temp2=series_temp[t_ini_ts:t_fim_ts]
        #
        x_train = series_temp1.iloc[:,0:N_REGRESS]
        y_train = series_temp1.iloc[:,N_REGRESS:N_REGRESS + MAX_STEPS]
        x_test = series_temp2.iloc[:,0:N_REGRESS]
        y_test = series_temp2.iloc[:,N_REGRESS:N_REGRESS + MAX_STEPS]
        
        if training:
         values2=np.concatenate((x_train.values,y_train.values),axis=1)
        else:
         values2=np.concatenate((x_test.values,y_test.values),axis=1)
		
        return M4Dataset(#ids=m4_info.M4id.values,
                         #groups=m4_info.SP.values,
                         #frequencies=m4_info.Frequency.values,
                         #horizons=m4_info.Horizon.values,
						 values=values2)
                         #values=np.load(
                         #    train_cache_file if training else test_cache_file,
                         #    allow_pickle=True))


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }  # different predict length
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
        'Yearly': 1.5,
        'Quarterly': 1.5,
        'Monthly': 1.5,
        'Weekly': 10,
        'Daily': 10,
        'Hourly': 10
    }  # from interpretable.gin


def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)
