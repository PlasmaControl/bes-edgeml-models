try:
    from ..main.train_base import _Trainer
except ImportError:
    from bes_ml.main.train_base import _Model


class Confinement_Classification_Model(_Trainer):

    def __init__(self) -> None:
        pass


if __name__=='__main__':
    Confinement_Classification_Model()