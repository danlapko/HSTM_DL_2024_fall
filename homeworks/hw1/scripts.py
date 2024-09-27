import random
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    """Для установки ceed"""
    random.seed(seed)  # Для стандартной библиотеки Python
    np.random.seed(seed)  # Для NumPy
    torch.manual_seed(seed)  # Для PyTorch на CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Для PyTorch на GPU
        torch.cuda.manual_seed_all(seed)  # Если используется несколько GPU
    torch.backends.cudnn.deterministic = True  # Гарантирует детерминированное поведение
    torch.backends.cudnn.benchmark = False  # Отключение оптимизаций для повышения стабильности


def split_df(df: pd.DataFrame, target_name: str, useless_columns=None) -> Tuple[pd.DataFrame, pd.Series]:
    """Метод для разделение pd.DataFrame на X, y"""

    useless_columns = useless_columns if useless_columns else []

    X = df.drop([target_name] + useless_columns, axis=1)
    y = df[target_name]

    return X, y


def convert_to_tensor(X: Union[pd.DataFrame, np.array], y: Union[pd.Series, np.array] = None):
    """Метод для конвертации pd.DataFrame в torch.tensor"""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values

    # Если y указан
    if y is not None:
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).long()
    # Если y не указан
    return torch.tensor(X, dtype=torch.float32)
