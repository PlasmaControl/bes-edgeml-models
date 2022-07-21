try:
    from ..main.train_base import _Trainer
except ImportError:
    from bes_models_2.main.train_base import _Model


class ELM_Regression_Model(_Trainer):

    def __init__(self) -> None:
        pass


if __name__=='__main__':
    ELM_Regression_Model()