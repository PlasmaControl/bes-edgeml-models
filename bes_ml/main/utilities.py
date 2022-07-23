import inspect
import logging

def _print_class_parameters(
    cls = None,
    locals_copy: dict = None,
    logger: logging.Logger = None,
) -> None:
    # print kwargs from __init__
    logger.info(f"Class `{cls.__name__}` parameters:")
    class_parameters = inspect.signature(cls).parameters
    for p_name in class_parameters:
        local_value = locals_copy[p_name]
        default_value = class_parameters[p_name].default
        if isinstance(local_value, dict):
            local_value = default_value = '<dict>'
        if local_value == default_value:
            logger.info(f"  {p_name:24s}:  {local_value}")
        else:
            logger.info(f"  {p_name:24s}:  {local_value}  (default {default_value})")

