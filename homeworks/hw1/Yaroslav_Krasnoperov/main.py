import argparse

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

torch.manual_seed(17)


def load_data(train_csv, val_csv, test_csv):

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    X_train = df_train.drop(["order0", "order1", "order2"], axis=1)
    y_train = df_train["order0"]

    X_val = df_val.drop(["order0", "order1", "order2"], axis=1)
    y_val = df_val["order0"]

    X_test = df_test

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.int64)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_val, y_val, X_test


class ClassificationNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(360, 360)
        self.fc2 = nn.Linear(360, 1080)
        self.fc3 = nn.Linear(1080, 720)
        self.fc4 = nn.Linear(720, 540)
        self.fc5 = nn.Linear(540, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)

        return x


def init_model(lr: float):
    model = ClassificationNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def evaluate(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=1)
        accuracy = accuracy_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)

    return predictions, accuracy, conf_matrix


def predict(model, X):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=1)

    return predictions


def train(
    model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs, batch_size
):
    for epoch in range(epochs):
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= X_train.size(0)
        print(f"\t Train: Epoch {epoch}, train Loss: {train_loss}")

        predictions, val_accuracy, conf_matrix = evaluate(model, X_val, y_val)
        print(f"val Accuracy {val_accuracy}")

    return model


def main(args):
    X_train, y_train, X_val, y_val, X_test = load_data(
        args.train_csv, args.val_csv, args.test_csv
    )
    model, criterion, optimizer = init_model(args.lr)
    model = train(
        model,
        criterion,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        args.num_epoches,
        args.batch_size,
    )
    predictions = predict(model, X_test)

    pd.DataFrame(predictions).to_csv("submission.csv")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", default="../data/train.csv")
    parser.add_argument("--val_csv", default="../data/val.csv")
    parser.add_argument("--test_csv", default="../data/test.csv")
    parser.add_argument("--out_csv", default="submission.csv")
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--batch_size", default=1024)
    parser.add_argument("--num_epoches", default=10)

    args = parser.parse_args()
    main(args)
