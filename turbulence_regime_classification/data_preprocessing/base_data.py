import numpy as np
import pandas as pd
import os
import re


class BaseData:
    def __init__(self, args, logger):
        self.args = args

        self.data_dir = self.args.input_data_dir
        assert (os.path.exists(self.data_dir))

        self.logger = logger

        self.logger.info(f'-------->  Data directory: {self.data_dir}')
        self.logger.info(f'-------->  {len(os.listdir(self.data_dir))} files!')

        self.elm_indices = [int(re.findall(r'_(\d+).hdf5', s)[0]) for s in os.listdir(self.data_dir)]

        return

    def get_data(self):
        pass

    def read_excel(self):
        pass

    def read_csv(self, db_type):
        raise NotImplementedError('Unable to read CSV at this time.')


