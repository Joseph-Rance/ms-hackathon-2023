import torch

from environment import Environment
from model import FullyConnected
from train import train, predict
from visualisation import visualise_data_processed

def main():

    DEVICE = "cpu"
    NUM_EPISODES = 500

    config = {
        "lr": 0.001,
        "momentum": 0.9,
        "reward_weights": (100, 2, 1),
        "gamma": 1,
    }

    print("constructing model")
    model = FullyConnected().to(DEVICE)

    print("constructing environment")
    environment = Environment()

    print("training model")
    losses = train(environment, model, NUM_EPISODES, config, device=DEVICE)

    # example run

    BASELINE_VMS = 1

    metrics_baseline = {
        "availability": [],
        "num_vms": [BASELINE_VMS]*24,
        "data_to_process": [],
        "data_processed": [],
        "reward": []
    }
    metrics_rl = deepcopy(metrics_baseline)

    print("constructing test environment")
    environment_baseline = Environment()
    environment_rl = Environment()

    environment_rl.a = environment_baseline.a  # make sure this is a fair test

    print("running test")
    for _ in range(24):
        environment_baseline.step(BASELINE_VMS)  # baseline run with constant number of VMs
        metrics_baseline["availability"].append(environment_baseline.get_availability())
        metrics_baseline["data_to_process"].append(environment_baseline.get_data_processed())
        metrics_baseline["data_processed"].append(environment_baseline.get_data_to_process())
        metrics_baseline["reward"].append(environment_baseline.get_reward(*config["reward_weights"]))

        s = environment_rl.get_state_vector()
        with torch.no_grad():
            num_vms = predict(model, s, action_space=range(20))
        environment_rl.step(num_vms)  # baseline run with constant number of VMs
        metrics_rl["availability"].append(environment_rl.get_availability())
        metrics_rl["num_vms"].append(num_vms)
        metrics_rl["data_to_process"].append(environment_rl.get_data_processed())
        metrics_rl["data_processed"].append(environment_rl.get_data_to_process())
        metrics_rl["reward"].append(environment_rl.get_reward(*config["reward_weights"]))
    
    # plot graphs & compute environmental benefits!

    # TODO: see comment above - use metrics_baseline & metrics_rl

    print("losses:", losses)
    print("temp results:", metrics_baseline, metrics_rl)

    visualise_data_processed(metrics_rl["data_processed"])

if __name__ == "__main__":

    from copy import deepcopy
    main()