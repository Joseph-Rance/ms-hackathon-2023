from environment import Environment
from model import FullyConnected
from train import train, predict
from visualisation import visualise_data_processed

def main():

    DEVICE = "cpu"

    config = {
        "num_updates": 5000,
        "episodes_per_update": 10,
        "batch_size": 24*10,
        "reward_weights": (2, 1, 1000),  # on peak, off peak, completion
        "gamma": 0.25,
        "epsilon": 1.0,
        "epsilon_decay": 0.99
    }

    print("constructing model")
    model = FullyConnected().to(DEVICE)

    print("constructing environment")
    environment = Environment()

    print("training model")
    losses, rewards = train(environment, model, config, device=DEVICE)

    SMOOTHING = 30
    smooth_losses, smooth_rewards = [], []
    for i in range(len(losses) - SMOOTHING):
        smooth_losses.append(sum(losses[i:i+SMOOTHING]))
        smooth_rewards.append(sum(rewards[i:i+SMOOTHING]))

    # example run

    BASELINE_VMS = 2

    metrics_baseline = {
        "availability": [],
        "req_num_vms": [],
        "prov_num_vms": [],
        "data_to_process": [],
        "data_processed": [],
        "reward": []
    }
    metrics_rl = deepcopy(metrics_baseline)
    metrics_baseline["req_num_vms"] = [BASELINE_VMS]*24

    print("constructing test environment")
    environment_baseline = Environment()
    environment_rl = Environment()

    environment_rl.a = environment_baseline.a  # make sure this is a fair test

    print("running test")
    for _ in range(24):
        environment_baseline.step(BASELINE_VMS)  # baseline run with constant number of VMs
        metrics_baseline["availability"].append(environment_baseline.get_availability())
        metrics_baseline["prov_num_vms"].append(environment_baseline.get_current_vms())
        metrics_baseline["data_to_process"].append(environment_baseline.get_data_to_process())
        metrics_baseline["data_processed"].append(environment_baseline.get_data_processed())
        metrics_baseline["reward"].append(environment_baseline.get_reward(*config["reward_weights"]))

        s = environment_rl.get_state_vector()
        with torch.no_grad():
            num_vms = predict(model, s, action_space=range(5))
        environment_rl.step(num_vms)  # baseline run with constant number of VMs
        metrics_rl["availability"].append(environment_rl.get_availability())
        metrics_rl["req_num_vms"].append(num_vms)
        metrics_rl["prov_num_vms"].append(environment_rl.get_current_vms())
        metrics_rl["data_to_process"].append(environment_rl.get_data_to_process())
        metrics_rl["data_processed"].append(environment_rl.get_data_processed())
        metrics_rl["reward"].append(environment_rl.get_reward(*config["reward_weights"]))

    # plot graphs & compute environmental benefits

    print("results:", metrics_baseline, metrics_rl)  # TEMP

    visualise_data_processed(metrics_rl["data_processed"])

    plt.plot(smooth_rewards)
    plt.title('reward over time')
    plt.savefig("graphs/rewards.png")
    plt.close()
    plt.plot([log(i) for i in smooth_losses])
    plt.title('loss over time')
    plt.savefig("graphs/losses.png")
    plt.close()

if __name__ == "__main__":

    from math import log
    from copy import deepcopy
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    SEED = 0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    main()