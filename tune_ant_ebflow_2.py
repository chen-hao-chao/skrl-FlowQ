import ray
from ray import tune
import os
from torch_ant_ebflow_off import _train
from skrl.agents.torch.ebflow import EBFlow_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

def trainer(tuner):
    id = tuner['id']
    grad_clip = tuner['grad_clip']
    tau = tuner['tau']
    alpha = tuner['alpha']
    lr = tuner['lr']
    bs = tuner['bs']
    num_envs = tuner['num_envs']
    timesteps = tuner['timesteps']
    path = tuner['path']

    # 
    description = path + "(id="+ str(id)+")" + \
                    "(lr="+ str(lr)+")" + \
                    "(bs="+ str(bs)+")" + \
                    "(envs="+ str(num_envs)+")" + \
                    "(ts="+ str(timesteps)+")" + \
                    "(gc="+ str(grad_clip)+")" + \
                    "(tau="+ str(tau)+")" + \
                    "(alpha="+ str(alpha)+")"
    
    # rewrite base config
    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = EBFlow_DEFAULT_CONFIG.copy()
    cfg["polyak"] = tau
    cfg["entropy_value"] = alpha
    cfg["grad_norm_clip"] = grad_clip
    cfg["learning_rate"] = lr
    cfg["batch_size"] = bs
    cfg["num_envs"] = num_envs
    cfg["timesteps"] = timesteps
    cfg["experiment"]["directory"] = description
    # --------
    cfg["gradient_steps"] = 1
    cfg["discount_factor"] = 0.99    
    cfg["random_timesteps"] = 100
    cfg["learning_starts"] = 100
    cfg["state_preprocessor"] = RunningStandardScaler
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 1000
    cfg["experiment"]["checkpoint_interval"] = 10000
    
    _train(cfg)

# ====================================

def main():
    ray.init(num_gpus=8) # 8
    
    search_space = {
        "grad_clip": tune.grid_search([30]),
        "tau": tune.grid_search([0.01, 0.005]),
        "alpha": tune.grid_search([0.5, 0.2, 0.1, 0.05, 0.005]),
        "lr": tune.grid_search([5e-4, 1e-3]),
        "bs": tune.grid_search([2048, 4096]),
        "num_envs": tune.grid_search([128]),
        "timesteps": tune.grid_search([500000]),
        "id": tune.grid_search([0]),
        "path": tune.grid_search(["/workspace/skrl-FlowQ/runs/results/"]),
    }
    
    analysis = tune.run(
        trainer, 
        num_samples=1,
        resources_per_trial={'cpu': 1, 'gpu': 0.2}, # 0.2
        config=search_space,
    )

if __name__ == '__main__':
    main()