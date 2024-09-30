import argparse

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

BATCH_SIZE = 1024

torch.manual_seed(42)


class NeuralClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.LeakyReLU = torch.nn.LeakyReLU()
        self.fc1 = nn.Linear(360, 180)
        self.fc2 = nn.Linear(180, 90)
        self.fc3 = nn.Linear(90, 3)

    def forward(self, data):
        data = self.LeakyReLU(self.fc1(data))
        data = self.LeakyReLU(self.fc2(data))
        data = self.LeakyReLU(self.fc3(data))
        return data


def load_data(train_csv: str, val_csv: str, test_csv: str):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    y_train = train_df.iloc[:, -3]
    X_train = train_df.iloc[:, 0:360]

    y_val = val_df.iloc[:, -3]
    X_val = val_df.iloc[:, 0:360]

    X_test = test_df.iloc[:, 0:360]

    return X_train, y_train, X_val, y_val, X_test


def preprocess_data(X_train, y_train, X_val, y_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_train = torch.tensor(X_train, dtype=torch.float32)

    y_val = torch.tensor(y_val, dtype=torch.int64)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test


def init_model():
    model = NeuralClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model, criterion, optimizer


def evaluate(model, X, y):
    with torch.no_grad():
        predictions = model(X).argmax(dim=1)
        accuracy = accuracy_score(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)
    return predictions, accuracy, conf_matrix


def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs):
    for epoch in range(epochs+1):
        optimizer.zero_grad()
        for i in range(0, X_train.size(0), BATCH_SIZE):
            X_batch = X_train[i:i + BATCH_SIZE]
            y_batch = y_train[i:i + BATCH_SIZE]
            y = model(X_batch)
            loss = criterion(y, y_batch)
            loss.backward()
            optimizer.step()

        predictions, accuracy, conf_matrix = evaluate(model, X_val, y_val)
        if epoch % 10 == 0:
            print(f'\nepoch: {epoch}\naccuracy: {accuracy}, conf_matrix: \n{conf_matrix}')
    return model


def predict(X, model):
    with torch.no_grad():
        model.eval()
        predictions = model(X).argmax(dim=1)
    return predictions


def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test = load_data(
        '../data/train.csv', '../data/val.csv', '../data/test.csv')

    # Preprocess data
    X_train, y_train, X_val, y_val, X_test = preprocess_data(X_train, y_train, X_val, y_val, X_test)

    # Initialize model
    model, criterion, optimizer = init_model()

    # Train model
    model = train(model, criterion, optimizer, X_train, y_train, X_val, y_val, 100)

    # Predict on test set
    predictions = predict(X_test, model)

    # dump predictions to 'submission.csv'
    pd.DataFrame(predictions).to_csv('submission.csv')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--train_csv', default='homeworks/hw1/data/train.csv')
#     parser.add_argument('--val_csv', default='homeworks/hw1/data/val.csv')
#     parser.add_argument('--test_csv', default='homeworks/hw1/data/test.csv')
#     parser.add_argument('--out_csv', default='homeworks/hw1/data/submission.csv')
#     parser.add_argument('--lr', default=0)
#     parser.add_argument('--batch_size', default=0)
#     parser.add_argument('--num_epoches', default=0)
#
#     args = parser.parse_args()
#     main(args)

main()
