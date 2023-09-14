def train(environment, model):
    # 0. simulate a full run using our model and the environment
    #     - for each timestep (hour) we want to store the current state and the reward
    # 1. compute the "value" of each state based on the rewards we stored
    # 2. train the model to map from state -> reward

    # this is the optimiser we will use to train
    optimiser = SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()  # put the model in "training mode"

    for episode in range(num_episodes):  # each iteration is one day simulation

        dataset = # simulate to get dataset of tuple for each timestep (state + #VMs, value)

        # this allows us to efficiently load the data into our model
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        epoch_loss = total = correct = 0
        for x, y in loader:  # train on each pair of input, value in the loader
            x, y = x.to("cuda"), y.to("cuda")  # (for )

            optimiser.zero_grad()

            z = self.model(x)  # get model's prediction
            loss = F.cross_entropy(z, y)  # get the loss between the model's prediction and the true value

            loss.backward()
            optimiser.step()  # update the model based on the loss

def predict():
    # inputs:
    #  - current state
    # output:
    #  - how many VMs to use
    #
    # v = -infinity
    # 1. loop over each number of VMs (0, 1, 2, ... num_available)
    # 2.    v <- max(v, M(state, number of VMs))
    # 3. return the number of VMs that had the maximum v