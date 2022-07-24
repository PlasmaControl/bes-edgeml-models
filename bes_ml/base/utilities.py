from typing import Union, Type
from pathlib import Path
import inspect
import logging

import yaml

def _print_inputs(
    cls: Type = None,
    locals_copy: dict = None,
    logger: logging.Logger = None,
) -> None:
    # print kwargs from __init__
    logger.info(f"Class `{cls.__name__}` parameters:")
    class_parameters = inspect.signature(cls).parameters
    for p_name in class_parameters:
        if p_name == 'logger': continue
        local_value = locals_copy[p_name]
        default_value = class_parameters[p_name].default
        if isinstance(local_value, dict):
            local_value = default_value = '<dict>'
        if local_value == default_value:
            logger.info(f"  {p_name:24s}:  {local_value}")
        else:
            logger.info(f"  {p_name:24s}:  {local_value}  (default {default_value})")

def _save_inputs_to_yaml(
    cls: Type = None, 
    locals_copy: dict = None,
    filename: Union[str,Path] = None,
) -> None:
    """
    Save locals from __init__() call to yaml.
    """
    filename = Path(filename)
    parameters = inspect.signature(cls).parameters
    inputs = {}
    for p_name in parameters:
        if p_name == 'logger': continue
        value = locals_copy[p_name]
        inputs[p_name] = value if not isinstance(value, Path) else value.as_posix()
    with filename.open('w') as parameters_file:
        yaml.safe_dump(
            inputs,
            parameters_file,
            default_flow_style=False,
            sort_keys=False,
        )

