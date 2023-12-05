from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

path = Path(os.getcwd()).parent
df = Path(str(path) + '/data/extracted_df.csv')

data = pd.read_csv(df)
df_to_work = data[['expression', 'AU01', 'AU02', 'AU04',
                   'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
                   'AU11', 'AU12', 'AU14', 'AU15', 'AU17',
                   'AU20', 'AU23', 'AU24', 'AU25', 'AU26',
                   'AU28', 'AU43']]


class MLP(nn.Module):
    def __init__(self, features_in=2, features_out=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features_in, 100),
            nn.ReLU(),
            nn.Linear(100, features_out),
        )

    def forward(self, input):
        return self.net(input)


class MultiEmoVA(Dataset):
    def __init__(self, data):
        super().__init__()

        # everything in pytorch needs to be a tensor
        self.inputs = torch.tensor(data.drop("expression", axis=1).to_numpy(dtype=np.float32))

        # we need to transform label (str) to a number. In sklearn, this is done internally
        self.index2label = [label for label in data["expression"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}

        self.labels = torch.tensor(data["expression"].apply(lambda x: torch.tensor(label2index[x])))

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)


dataset = MultiEmoVA(df_to_work)
K = 10
# passing a generator to random_split is similar to specifying the seed in sklearn
generator = torch.Generator().manual_seed(2023)
# we need to move our model to the correct device
cross_validation = []
# it is common to do a training loop multiple times, we call these 'epochs'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

for k in tqdm(range(K), position=0, leave=True):
    train, test = random_split(dataset, [0.8, 0.2], generator=generator)

    print("Number of objects in Training set: ", len(train))
    print("Number of objects in Validation set: ", len(test))

    train_loader = DataLoader(train, batch_size=11, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    model = MLP(train[0][0].shape[0], len(dataset.index2label)).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    max_epochs = 200
    for epoch in tqdm(range(max_epochs), position=0, leave=True):
        for inputs, labels in train_loader:
            # both input, output and model need to be on the same device
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = model(inputs)
            loss = loss_fn(out, labels)

            loss.backward()
            optim.step()
            optim.zero_grad()

    # print(f"Training epoch {epoch} average loss: {loss:.4f}")
    # tell pytorch we're not training anymore
    with torch.no_grad():
        test_loader = DataLoader(test, batch_size=4)
        correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            # Here we go from the models output to a single class and compare to ground truth
            correct += (predictions.softmax(dim=1).argmax(dim=1) == labels).sum()
        print(f"Accuracy is: {correct / len(test) * 100:0.4f}%")
    k_run_accuracy = correct / len(test) * 100
    cross_validation.append(k_run_accuracy)
print(f"Mean accuracy: {sum(cross_validation) / len(cross_validation):0.4f}%")
print("Finished training")
