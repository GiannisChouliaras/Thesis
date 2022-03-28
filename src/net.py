from statistics import mean
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    """Class represents our Neural Network"""

    def __init__(self, inFeats: int, outFeats: int, fHidden: int, sHidden: int) -> None:
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=inFeats, out_features=fHidden),
            nn.Tanh(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=fHidden, out_features=sHidden),
            nn.Tanh(),
            nn.Linear(in_features=sHidden, out_features=outFeats),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def main(save_model=False) -> None:

    # HyperParameters
    LR = 0.01
    ITERATIONS = 200
    INPUT = 3
    OUTPUT = 1
    H1 = 10
    H2 = 5

    # import the numpy arrays
    data = np.load("../data/arrays/data.npy")
    target = np.load("../data/arrays/target.npy")

    # convert arrays to tensors
    data = torch.tensor(data, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    # K-Fold
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # ************************************
    #       Starting Training loop
    # ************************************
    fold = 0
    train_loss_lst: List = []
    valid_loss_lst: List = []

    for trainSet, testSet in k_fold.split(data):
        fold += 1
        writer = SummaryWriter(f"../runs/loss/fold{fold}")
        X_train, X_test = data[trainSet], data[testSet]
        y_train, y_test = target[trainSet], target[testSet]

        # reshape tensors
        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        # init model, loss and optimizer
        model = Net(inFeats=INPUT, outFeats=OUTPUT, fHidden=H1, sHidden=H2)
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=10)

        # starting the train procedure
        step = 0
        for epoch in range(ITERATIONS):
            prediction = model(X_train)
            training_loss = loss(prediction, y_train)
            optimizer.zero_grad()
            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # validation loss
            with torch.no_grad():
                model.eval()
                valid_prediction = model(X_test)
                validation_loss = loss(valid_prediction, y_test)
                model.train()

            scheduler.step(metrics=validation_loss)
            # print(optimizer.state_dict()['param_groups'][0]['lr'])

            writer.add_scalar(tag="training loss", scalar_value=training_loss, global_step=step)
            writer.add_scalar(tag="validation loss", scalar_value=validation_loss, global_step=step)
            step += 1

        # add training and validation lost in lists
        train_loss_lst.append(round(training_loss.item(), 3))
        valid_loss_lst.append(round(validation_loss.item(), 3))

        del model

    print(f"Average: Training: {mean(train_loss_lst):.3f} --- Validation: {mean(valid_loss_lst):.3f}")

    if not save_model:
        return

    # Create and train the final model after the cross validation
    model = Net(inFeats=INPUT, outFeats=OUTPUT, fHidden=H1, sHidden=H2)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # reshape target
    target = target.reshape(target.shape[0], 1)

    # training before saving
    for epoch in range(ITERATIONS):
        prediction = model(data)
        training_loss = loss(prediction, target)
        optimizer.zero_grad()
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print(f"The final training loss is {training_loss.item():.3f}")
    print(model)
    print(f"Got {sum([param.nelement() for param in model.parameters()])} parameters")
    torch.save(model.state_dict(), "../models/net.pt")


if __name__ == "__main__":
    main(save_model=False)
