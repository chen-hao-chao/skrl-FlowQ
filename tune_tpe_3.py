import ray
from ray import tune
import os
from trainer_sac import _train
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

def trainer(tuner):
    id = tuner['id']
    grad_clip = tuner['grad_clip']
    tau = tuner['tau']
    alpha = tuner['alpha']
    lr = tuner['lr']
    loading = tuner['loading']
    num_envs = tuner['num_envs']
    bs = int(loading / num_envs)
    timesteps = tuner['timesteps']
    path = tuner['path']
    task_name = tuner['task_name']

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
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["task_name"] = task_name
    cfg["polyak"] = tau
    cfg["initial_entropy_value"] = alpha
    cfg["grad_norm_clip"] = grad_clip
    cfg["actor_learning_rate"] = lr
    cfg["critic_learning_rate"] = lr
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
    cfg["memory_size"] = 15000
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 5000
    cfg["experiment"]["checkpoint_interval"] = timesteps
    
    _train(cfg)

# ====================================

def main():
    ray.init(num_gpus=8)
    
    search_space = {
        "task_name": tune.grid_search(["AllegroHand"]),
        "grad_clip": tune.grid_search([0]),
        "tau": tune.grid_search([0.0025, 0.001]),
        "alpha": tune.grid_search([0.1, 0.075, 0.05, 0.01]),
        "lr": tune.grid_search([3e-4]),
        "loading": tune.grid_search([131072]),
        "num_envs": tune.grid_search([512]),
        "timesteps": tune.grid_search([1000000]),
        "id": tune.grid_search([0,1,2,3,4]),
        "path": tune.grid_search(["/mnt/nfs/skrl-FlowQ/runs/results_sac_allegrohand_baseline/"]),
    }
    
    analysis = tune.run(
        trainer, 
        num_samples=1,
        resources_per_trial={'cpu': 4, 'gpu': 0.2},
        config=search_space,
    )

if __name__ == '__main__':
    main()