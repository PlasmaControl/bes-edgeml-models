from .base_arguments import BaseArguments


class TestArguments(BaseArguments):
    """Command line arguments for testing. It inherits the BaseArguments for
    common parameters.
    """

    def create_parser(self, parser):
        parser = BaseArguments.create_parser(self, parser)
        # test parameters

        parser.add_argument(
            "--test_data_info",
            action="store_false",
            default=False,
            help="if true, show the information about the test data.",
        )
        parser.add_argument(
            "--plot_num",
            type=int,
            default=12,
            help="number of elm events to plot to show the predictions, [6 | 12].",
        )
        parser.add_argument(
            "--num_rows",
            type=int,
            default=4,
            help="number of rows in the plot to show the predictions, [4 | 3].",
        )
        parser.add_argument(
            "--num_cols",
            type=int,
            default=3,
            help="number of columns in the plot to show the predictions, [3 | 2].",
        )
        parser.add_argument(
            "--plot_data",
            action="store_true",
            default=False,
            help="if true, randomly select `plot_num` elm events with ground truth and predictions.",
        )
        parser.add_argument(
            "--show_metrics",
            action="store_true",
            default=False,
            help="if true, show and save classfication report, roc details and "
            "confusion matrix.",
        )

        self.is_train = False

        return parser
