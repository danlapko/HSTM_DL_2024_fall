import os
from enum import Enum

root_path = os.getcwd()
target = "order0"
useless_columns = ["order1", "order2"]

# Параметры модели
learning_rate = 0.00010520007192097682
num_epoches = 71
batch_size = 1024

model_params = {
    "first_layer_size": 512,
    "second_layes_size": 128,
    "in_features": 360,
    "out_features": 3,
}


class Paths(str, Enum):
    """Enum с путями"""

    train_path: str = os.path.join(root_path, "homeworks", "hw1", "data", "train.csv")
    val_path: str = os.path.join(root_path, "homeworks", "hw1", "data", "val.csv")
    test_path: str = os.path.join(root_path, "homeworks", "hw1", "data", "test.csv")
