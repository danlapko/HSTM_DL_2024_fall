import torch
import torch.nn as nn


class MLP(nn.Module):
    """Класс - архитектура модели нейронной сети"""

    def __init__(self, model_params: dict[str, int]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(model_params["in_features"], model_params["first_layer_size"])
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(model_params["first_layer_size"])
        self.fc2 = nn.Linear(model_params["first_layer_size"], model_params["second_layes_size"])
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(model_params["second_layes_size"])
        self.fc3 = nn.Linear(model_params["second_layes_size"], model_params["out_features"])

    def forward(self, x):
        """Метод для прогона по слоям модели"""

        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        """Метод для инференса"""

        with torch.no_grad():
            x = self.forward(x)  # Логиты
            probabilities = torch.softmax(x, dim=1)  # Применяем softmax
            predicted_classes = torch.argmax(probabilities, dim=1)  # Применяем argmax для получения класса
        return predicted_classes
