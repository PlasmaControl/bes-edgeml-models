import inspect
import logging

def _get_class_parameters_and_defaults(
    cls = None,
) -> dict:
    signature = inspect.signature(cls)
    parameters_and_defaults = {parameter.name: parameter.default 
        for parameter in signature.parameters.values()}
    return parameters_and_defaults

def _print_kwargs(
    cls = None,
    locals_copy: dict = None,
    logger: logging.Logger = None,
) -> None:
    # print kwargs from __init__
    logger.info(f"Class `{cls.__name__}` parameters:")
    class_parameters = _get_class_parameters_and_defaults(cls)
    for key, default_value in class_parameters.items():
        if key == 'kwargs': continue
        value = locals_copy[key]
        if value == default_value:
            logger.info(f"  {key:22s}:  {value}")
        else:
            logger.info(f"  {key:22s}:  {value} (default {default_value})")

