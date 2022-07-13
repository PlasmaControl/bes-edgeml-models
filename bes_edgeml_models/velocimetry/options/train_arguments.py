from typing import Union

from bes_edgeml_models.base.options.train_arguments import TrainArguments as PredictionArguments

import argparse
from pathlib import Path

class TrainArguments(PredictionArguments):

    def create_parser(self, parser: argparse.ArgumentParser):
        parser = PredictionArguments.create_parser(self, parser)

        # training parameters
        parser.add_argument(
            "--input_data_dir",
            type=str,
            default=str(Path(__file__).parents[1] / 'data'),
            help="Location of HDF5 data sets. If absolute path, that path is used. If relative, path is relative to base_dir.",
        )

        parser.add_argument(
            "--dataset_to_ram",
            action="store_true",
            default=False,
            help="Loads entire directory of datasets into RAM for faster access."
        )

        parser.is_train = True

        return parser

    def parse(
        self,
        arg_list: Union[list, None] = None,
        existing_namespace=None,
    ):
        """
        Parse arguments
        Note: `arg_list` will override command line inputs from `sys.argv`
        """
        args = self._gather_args(
            arg_list=arg_list,
            existing_namespace=existing_namespace,
            )

        # Set unused args
        args.input_data_file = None
        args.label_look_ahead = 0
        args.truncate_buffer = 0

        self.args = args

        return args
