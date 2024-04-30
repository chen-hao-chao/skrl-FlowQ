import ray
from ray import tune
import os
from trainer_ebflow import _test
from skrl.agents.torch.ebflow import EBFlow_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

def main(env_id):
    cfg = EBFlow_DEFAULT_CONFIG.copy()
    if env_id == "AllegroHand":
        cfg["task_name"] = "AllegroHand"
        cfg["polyak"] = 0.001
        cfg["entropy_value"] = 0.1
        cfg["grad_norm_clip"] = 30
        cfg["learning_rate"] = 1e-3
        cfg["batch_size"] = int(131072 / 512)
        cfg["num_envs"] = 512
        cfg["timesteps"] = 1000000
        cfg["random_timesteps"] = 100
        cfg["sigma_max"] = -0.3
        cfg["sigma_min"] = -4.0
        cfg["experiment"]["directory"] = "_results_isaac_compare/standard/Allegro/meow/1/24-04-09_06-24-34-491494_EBFlow/checkpoints/best_agent.pt" # checkpoint
    elif env_id == "Ant":
        cfg["task_name"] = "Ant"
        cfg["polyak"] = 0.0005
        cfg["entropy_value"] = 0.075
        cfg["grad_norm_clip"] = 30
        cfg["learning_rate"] = 1e-3
        cfg["batch_size"] = int(131072 / 128)
        cfg["num_envs"] = 128
        cfg["timesteps"] = 1000000
        cfg["random_timesteps"] = 100
        cfg["sigma_max"] = 2.0
        cfg["sigma_min"] = -5.0
        cfg["experiment"]["directory"] = "_results_isaac_compare/standard/Ant/meow/1/24-04-09_06-24-19-851518_EBFlow/checkpoints/best_agent.pt" # checkpoint
    elif env_id == "Anymal":
        cfg["task_name"] = "Anymal"
        cfg["polyak"] = 0.025
        cfg["entropy_value"] = 0.00075
        cfg["grad_norm_clip"] = 30
        cfg["learning_rate"] = 1e-3
        cfg["batch_size"] = int(131072 / 128)
        cfg["num_envs"] = 128
        cfg["timesteps"] = 1000000
        cfg["random_timesteps"] = 100
        cfg["sigma_max"] = -1.0
        cfg["sigma_min"] = -5.0
        cfg["experiment"]["directory"] = "_results_isaac_compare/standard/Anymal/meow/1/24-04-10_08-34-49-488056_EBFlow/checkpoints/best_agent.pt" # checkpoint
    elif env_id == "FrankaCabinet":
        cfg["task_name"] = "FrankaCabinet"
        cfg["polyak"] = 0.01
        cfg["entropy_value"] = 0.075
        cfg["grad_norm_clip"] = 30
        cfg["learning_rate"] = 1e-3
        cfg["batch_size"] = int(131072 / 512)
        cfg["num_envs"] = 512
        cfg["timesteps"] = 1000000
        cfg["random_timesteps"] = 100
        cfg["sigma_max"] = -0.3
        cfg["sigma_min"] = -5.0
        cfg["experiment"]["directory"] = "_results_isaac_compare/standard/FrankaCabinet/meow/1/24-04-10_08-35-08-604878_EBFlow/checkpoints/best_agent.pt" # checkpoint
    elif env_id == "Humanoid":
        cfg["task_name"] = "Humanoid"
        cfg["polyak"] = 0.00025
        cfg["entropy_value"] = 0.25
        cfg["grad_norm_clip"] = 30
        cfg["learning_rate"] = 1e-3
        cfg["batch_size"] = int(131072 / 128)
        cfg["num_envs"] = 128
        cfg["timesteps"] = 1000000
        cfg["random_timesteps"] = 100
        cfg["sigma_max"] = -0.3
        cfg["sigma_min"] = -5.0
        cfg["experiment"]["directory"] = "_results_isaac_compare/standard/Humanoid/meow/1/24-04-09_07-06-48-041168_EBFlow/checkpoints/best_agent.pt" # checkpoint
    elif env_id == "Ingenuity":
        cfg["task_name"] = "Ingenuity"
        cfg["polyak"] = 0.0025
        cfg["entropy_value"] = 0.025
        cfg["grad_norm_clip"] = 30
        cfg["learning_rate"] = 1e-3
        cfg["batch_size"] = int(131072 / 128)
        cfg["num_envs"] = 128
        cfg["timesteps"] = 1000000
        cfg["random_timesteps"] = 100
        cfg["sigma_max"] = -0.3
        cfg["sigma_min"] = -4.0
        cfg["experiment"]["directory"] = "_results_isaac_compare/standard/Ingenuity/meow/1/24-04-09_12-07-20-651046_EBFlow/checkpoints/best_agent.pt" # checkpoint

    # --------
    cfg["gradient_steps"] = 1
    cfg["discount_factor"] = 0.99
    cfg["random_timesteps"] = 100
    cfg["learning_starts"] = 100
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["memory_size"] = 15000
    cfg["experiment"]["write_interval"] = 5000
    cfg["experiment"]["checkpoint_interval"] = 1000000
    
    _test(cfg)

# ===================================

if __name__ == '__main__':
    env_id = "AllegroHand"
    main(env_id)