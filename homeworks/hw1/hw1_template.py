import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from homeworks.hw1.config import (
    batch_size,
    learning_rate,
    model_params,
    num_epoches,
    target,
    useless_columns,
)
from homeworks.hw1.model import MLP
from homeworks.hw1.scripts import convert_to_tensor, set_seed, split_df

set_seed()


def load_data(train_csv_path: str, val_csv_path: str, test_csv_path: str):
    """Метод для загрузки всех датасетов и их разделения на X, y"""

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    X_train, y_train = split_df(df_train, target_name=target, useless_columns=useless_columns)
    X_val, y_val = split_df(df_val, target_name=target, useless_columns=useless_columns)
    X_test = pd.read_csv(test_csv_path)

    return X_train, y_train, X_val, y_val, X_test


def init_model(params: dict[str, int], lr: float):
    """Метод для инициализации модели, loss-функции и оптимизатора"""

    model = MLP(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    return model, criterion, optimizer


def evaluate(model, X, y):
    """Метод для инференса модели с известным Y"""

    model.eval()
    with torch.no_grad():
        predictions = model.predict(X).cpu().numpy()
        y = y.cpu().numpy()

        # Вычисляем матрицу ошибок
        accuracy = accuracy_score(y_true=y, y_pred=predictions)
        conf_matrix = confusion_matrix(y_true=y, y_pred=predictions)

    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, epochs, batch_size, X_val=None, y_val=None):
    """Метод для обучения модели и проверки его на валидационной выборке"""

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # Обновляем градиенты, forward, считаем loss и обновляем параметры
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= X_train.size(0) // batch_size
        train_losses.append(train_loss)

        if X_val is not None and y_val is not None:
            # Оценка на валидационной выборке
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, X_val.size(0), batch_size):
                    X_batch_val = X_val[i : i + batch_size]
                    y_batch_val = y_val[i : i + batch_size]

                    outputs_val = model(X_batch_val).squeeze()
                    loss_val = criterion(outputs_val, y_batch_val)
                    val_loss += loss_val.item()

            val_loss /= X_val.size(0) // batch_size  # Средний loss на валидационной выборке
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch}: Train Loss: {round(train_loss, 4)}"
                f"| Epoch {epoch}: Validation Loss: {round(val_loss, 4)}"
            )

    return train_losses, val_losses


def main(args):
    """Пайплайн для загрузки данных, обучения модели и ее инференса"""

    # Load data
    X_train, y_train, X_val, y_val, X_test = load_data(
        train_csv_path=args.train_csv, val_csv_path=args.val_csv, test_csv_path=args.test_csv
    )

    X_train, y_train = convert_to_tensor(X_train, y_train)
    X_val, y_val = convert_to_tensor(X_val, y_val)
    X_test = convert_to_tensor(X_test)

    # Initialize model
    model, criterion, optimizer = init_model(model_params, args.lr)

    # Train model
    _, _ = train(model, criterion, optimizer, X_train, y_train, args.num_epoches, args.batch_size, X_val, y_val)

    # Predict on val set
    predictions_val, accuracy_val, conf_matrix_val = evaluate(model, X_val, y_val)
    
    report = classification_report(y_true=y_val.cpu().numpy(), y_pred=predictions_val)
    print(report)
    print(f"accuracy: {accuracy_val}")
    print(f"conf_matrix_val: {conf_matrix_val}")

    # Predict on test set
    predictions = model.predict(X_test).cpu().numpy()

    # dump predictions to 'submission.csv'
    pd.Series(predictions).to_csv(args.out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", default="homeworks/hw1/data/train.csv")
    parser.add_argument("--val_csv", default="homeworks/hw1/data/val.csv")
    parser.add_argument("--test_csv", default="homeworks/hw1/data/test.csv")
    parser.add_argument("--out_csv", default="homeworks/hw1/submission.csv")
    parser.add_argument("--lr", default=learning_rate)
    parser.add_argument("--batch_size", default=batch_size)
    parser.add_argument("--num_epoches", default=num_epoches)

    args = parser.parse_args()
    main(args)
