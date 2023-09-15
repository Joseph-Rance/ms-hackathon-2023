from random import random, choice
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.data = (np.array(x), np.array(y).reshape((-1, 1)))

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

def train(environment, model, config, device="cpu"):
    # 0. simulate a full run using our model and the environment
    #     - for each timestep (hour) we want to store the current state and the reward
    # 1. compute the "value" of each state based on the rewards we stored
    # 2. train the model to map from state -> reward

    # this is the optimiser we will use to train
    optimiser = SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    model.train()  # put the model in "training mode"

    losses = []

    for _ in tqdm(range(config["num_episodes"])):  # each iteration is one day simulation

        # reward y[i] comes after state x[i]
        x, y = [], []

        environment.reset()
        # simulate to get dataset of tuple for each timestep (state + #VMs, value)
        for _ in range(24):
            s = environment.get_state_vector()
            if random() < config["epsilon"]:
                num_vms = choice(range(20))
            else:
                num_vms = predict(model, s, action_space=range(20), device=device)
            environment.step(num_vms)
            r = environment.get_reward(*config["reward_weights"])
            x.append(s + [num_vms])
            y.append(r)

        current = 0  # reduces weight of rewards that are further away
        for i in range(len(y)-1, -1, -1):
            current = current * config["gamma"] + y[i]
            y[i] = current

        dataset = SimpleDataset(x, y)

        # this allows us to efficiently load the data into our model
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        losses.append(0)

        for x, y in loader:  # train on each pair of input, value in the loader

            x, y = x.float().to(device), y.float().to(device)

            optimiser.zero_grad()

            z = model(x)  # get model's prediction

            loss = F.mse_loss(z, y)  # get the loss between the model's prediction and the true value
            losses[-1] += loss.item()

            loss.backward()
            optimiser.step()  # update the model based on the loss

    return losses

def predict(model, state, action_space, device="cpu"):
    # returns how many VMs to use
    #
    # v = -infinity
    # 1. loop over each number of VMs (0, 1, 2, ... num_available)
    # 2.    v <- max(v, M(state, number of VMs))
    # 3. return the number of VMs that had the maximum v

    best_value = -float("inf")
    best_action = [0]
    for a in action_space:
        value = model(torch.tensor(state + [a]).float().to(device))
        if best_value <= (val := value.item()):
            best_value = val
            best_action = a

    return best_action