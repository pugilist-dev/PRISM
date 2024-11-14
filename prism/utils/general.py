import toml
from loguru import logger
import os
import random

import numpy as np
import pandas as pd
import torch
import sklearn


def get_repo_name():
    try:
        with open('pyproject.toml', 'r') as file:
            pyproject_data = toml.load(file)
            project_name = pyproject_data.get('project', {}).get('name')
            return project_name
    except FileNotFoundError:
        logger.error("pyproject.toml file not found.")
        return "PROJECT_NAME"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "PROJECT_NAME"
    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sklearn.utils.check_random_state(seed)