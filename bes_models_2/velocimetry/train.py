try:
    from ..source.train_base import _Model_Trainer
except ImportError:
    from bes_models_2.source.train_base import _Model


class Velocimetry_Model(_Model_Trainer):

    def __init__(self) -> None:
        pass


if __name__=='__main__':
    Velocimetry_Model()