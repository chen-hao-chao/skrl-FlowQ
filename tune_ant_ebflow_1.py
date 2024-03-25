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

    # 
    description = "../../../../../results/"+ \
                    "(id="+ str(id)+")" + \
                    "(lr="+ str(lr)+")" + \
                    "(bs="+ str(bs)+")" + \
                    "(num_envs="+ str(num_envs)+")" + \
                    "(timesteps="+ str(timesteps)+")" + \
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
    cfg["learning_rate"] = lr #5e-4
    cfg["batch_size"] = bs #4096
    cfg["num_envs"] = num_envs # 64
    cfg["timesteps"] = timesteps #160000
    cfg["experiment"]["directory"] = description
    # --------
    
    _train(cfg)

# ====================================

def main():
    ray.init(num_gpus=3) # 8
    
    search_space = {
        "grad_clip": tune.grid_search([30]),
        "tau": tune.grid_search([0.01, 0.005]),
        "alpha": tune.grid_search([0.5, 0.2, 0.1, 0.05, 0.005]),

        "lr": tune.grid_search([5e-4, 1e-3]),
        "bs": tune.grid_search([2048, 4096]),
        "num_envs": tune.grid_search([64]),
        "timesteps": tune.grid_search([500000]),
        
        "id": tune.grid_search([0]),
    }
    
    dirpath = os.path.dirname(os.path.realpath(__file__))

    analysis = tune.run(
        trainer, 
        num_samples=1,
        local_dir="/workspace/skrl-FlowQ/runs/result_tuning_EBRL_isaac_ant", #os.path.join(dirpath, "runs/result_tuning_EBRL_isaac_ant"),
        resources_per_trial={'cpu': 1, 'gpu': 0.2}, # 0.2
        config=search_space,
    )

if __name__ == '__main__':
    main()