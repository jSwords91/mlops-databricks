import os
import pathlib
import yaml
import pprint
from typing import Dict, Any

def load_config(pipeline_name, verbose=False) -> Dict[str, Any]:
    """
    Utility function to use in Databricks notebooks to load the config yaml file for a given pipeline
    Return dict of specified config params

    Parameters
    ----------
    pipeline_name :  str
        Name of pipeline

    Returns
    -------
    Dictionary of config params
    """
    config_path = os.path.join('conf', f'{pipeline_name}.yml')
    
    try:
        pipeline_config = yaml.safe_load(pathlib.Path(os.pardir, config_path).read_text())
    except FileNotFoundError:
        try:
            pipeline_config = yaml.safe_load(pathlib.Path(config_path).read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the configuration file: {pipeline_name}.yml")
    
    if verbose:
        pprint.pprint(pipeline_config)

    return pipeline_config
