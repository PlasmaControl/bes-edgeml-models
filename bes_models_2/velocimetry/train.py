try:
    from ..main.train_base import _Trainer
except ImportError:
    from bes_models_2.main.train_base import _Model


class Velocimetry_Model(_Trainer):

    def __init__(self) -> None:
        pass


if __name__=='__main__':
    Velocimetry_Model()