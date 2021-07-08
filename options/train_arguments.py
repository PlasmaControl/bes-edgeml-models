from .base_arguments import BaseArguments


class TrainArguments(BaseArguments):
    """Command line arguments specifically used for training. This inherits the
    BaseArguments for all other common arguments.
    """

    def create_parser(self, parser):
        parser = BaseArguments.create_parser(self, parser)
        # training parameters
        parser.add_argument(
            "--focal_loss",
            action="store_true",
            default=False,
            help="if true, use focal loss instead of `BCELossWithLogits`.",
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="AdamW",
            help="optimizer to use, [SGD | Adam | AdamW].",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="learning rate.",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            help="hidden size of the RNN. Must be used "
            "if `use_rnn` is set to True.",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.005,
            help="L2 regularization factor.",
        )
        # training progress display parameters
        parser.add_argument(
            "--train_print_every",
            type=int,
            default=5000,
            help="frequency of showing training results on screen.",
        )
        parser.add_argument(
            "--valid_print_every",
            type=int,
            default=2000,
            help="frequency of showing validation results on screen.",
        )
        # lr scheduler parameters
        parser.add_argument(
            "--scheduler",
            type=str,
            default="CyclicLR",
            help="policy to change the learning rate, "
            "[CosineAnnealingLR | CyclicLR | ReduceLROnPlateau | "
            "OneCycleLR | ExponentialLR].",
        )
        parser.add_argument(
            "--cyclic_mode",
            type=str,
            default="triangular2",
            help="cycle policy with which the learning rate be changed."
            "Only applies when `CyclicLR` is used for lr scheduler, [triangular "
            "| triangular2 | exp_range].",
        )
        parser.add_argument(
            "--lr_plateau_mode",
            type=str,
            default="min",
            help="[min | max]. In `min` (`max`) mode, lr will be reduced when the "
            "quantity monitored (loss, accuracy, etc.) has stopped decreasing (increasing). "
            "Only used with `ReduceLROnPlateau` scheduler.",
        )
        parser.add_argument(
            "--decay_factor",
            type=float,
            default=0.5,
            help="multiplicative factor by which the learning rate be reduced, "
            "`new_lr = decay_factor * lr`. This factor also serves as the `gamma` "
            "value when `ExponentialLR` is used as the scheduler. Only used with "
            "`ReduceLROnPlateau` and `ExponentialLR` schedulers.",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=2,
            help="number of epochs with no improvement after which the learning "
            "rate will be reduced. Only used with `ReduceLROnPlateau` scheduler.",
        )
        parser.add_argument(
            "--T_max",
            type=int,
            default=10,
            help="maximum number of iterations for cosine annealing scheduler.",
        )
        parser.add_argument(
            "--eta_min",
            type=float,
            default=0,
            help="minimum learning rate for cosine annealing scheduler.",
        )
        parser.add_argument(
            "--base_lr",
            type=float,
            default=1e-4,
            help="initial learning rate which is the lower boundary for the cycle "
            "of the scheduler. Only used with `CyclicLR` scheduler.",
        )
        parser.add_argument(
            "--max_lr",
            type=float,
            default=1e-3,
            help="upper learning rate boundaries in the cycle of the scheduler "
            "Only used with `CyclicLR` and `OneCycleLR` scheduler.",
        )
        parser.add_argument(
            "--cycle_momentum",
            action="store_true",
            default=False,
            help="if true, momentum is cycled inversely to lr. Applies only to "
            "`CyclicLR` and must be set to False if `Adam` or `AdamW` are used as "
            "optimizers.",
        )
        parser.add_argument(
            "--pct_start",
            type=float,
            default=0.2,
            help="percentage of cycle (in number of steps) spent increasing the "
            "learning rate. Only used with `OneCycleLR` scheduler.",
        )
        parser.add_argument(
            "--anneal_policy",
            type=str,
            default="cos",
            help="specifies the annealing strategy. [cos | linear]: `cos` for cosine "
            "annealing, `linear` for linear annealing. Only used with `OneCycleLR` "
            "scheduler.",
        )
        self.is_train = True

        return parser
