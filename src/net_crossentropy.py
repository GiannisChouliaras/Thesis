import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold


class SoftNet(nn.Module):
    def __init__(
        self, inputs: int, outputs: int, hidden_size: int
    ) -> None:
        super(SoftNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=outputs),
        )

    def forward(self, x):
        return self.model(x)


def main(save_model=False) -> None:
    # HyperParameters
    LR = 1e-2
    ITERATIONS = 200
    INPUT = 3
    OUTPUT = 8
    HIDDEN = 10

    # import numpy arrays
    data = torch.tensor(np.load("../data/arrays/data.npy"), dtype=torch.float32)
    abstract = lambda t: t - 1
    targets = torch.tensor(abstract(np.load("../data/arrays/doctors.npy")))
    targets = targets.type(torch.LongTensor)

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    for trainSet, testSet in k_fold.split(data):
        X_train, X_test = data[trainSet], data[testSet]
        y_train, y_test = targets[trainSet], targets[testSet]

        model = SoftNet(
            inputs=INPUT,
            outputs=OUTPUT,
            hidden_size=HIDDEN
        )
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Training loop
        for epoch in range(ITERATIONS):
            prediction = model(X_train)
            training_loss = loss(prediction, y_train)
            optimizer.zero_grad()
            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                valid_prediction = model(X_test)
                validation_loss = loss(valid_prediction, y_test)
                model.train()

            if (epoch + 1) % 10 == 0:
                print(
                    f"epoch {epoch + 1}: training_loss: {training_loss.item():.3f} validation loss: {validation_loss.item():.3f}"
                )

        del model

    if not save_model:
        return

    model = SoftNet(inputs=INPUT, outputs=OUTPUT, hidden_size=HIDDEN)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(ITERATIONS):
        prediction = model(data)
        training_loss = loss(prediction, targets)
        optimizer.zero_grad()
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print(f"The final training loss is {training_loss.item():.3f}")
    torch.save(model.state_dict(), "../models/net_ce.pt")


if __name__ == "__main__":
    main(save_model=True)
