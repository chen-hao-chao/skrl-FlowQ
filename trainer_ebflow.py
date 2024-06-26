import os
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ebflow import EBFlow, EBFlow_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import FlowMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from normflows.nets import MLP
from normflows.transforms import Preprocessing
from normflows.distributions import MyConditionalDiagGaussian
from normflows.flows import MaskedCondAffineFlow, CondScaling, LULinearPermute
# define models (stochastic and deterministic models) using mixins
class Policy(FlowMixin, Model):
    def __init__(self, observation_space, action_space, device, alpha, sigma_max, sigma_min, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        FlowMixin.__init__(self, reduction)
        flows, q0 = self.init_Coupling(self.num_observations, self.num_actions, sigma_max=sigma_max, sigma_min=sigma_min)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.prior = q0.to(self.device)
        self.alpha = alpha
        self.unit_test(num_samples=10, scale=1)

    def init_Coupling(self, state_sizes, action_sizes, sigma_max=-0.3, sigma_min=-5):
        dropout_rate_flow = 0.1
        dropout_rate_scale = 0
        layer_norm_flow = True
        layer_norm_scale = False
        # sigma_max = -0.3
        # sigma_min = -5
        prior_transform = True # set True to perform additional tanh scaling for unlearnable prior.
        hidden_sizes = 64
        scale_hidden_sizes = 256
        hidden_layers = 2
        flow_layers = 2
        
        # Construct prior distribution
        prior_list = [state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
        loc = None
        log_scale = MLP(prior_list, init="zero")
        q0 = MyConditionalDiagGaussian(action_sizes, loc, log_scale, SIGMA_MIN=sigma_min, SIGMA_MAX=sigma_max, prior_transform=prior_transform)

        # Construct flow network
        flows = []
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_sizes)])
        for i in range(flow_layers):
            layers_list = [action_sizes+state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
            s = None
            t1 = MLP(layers_list, init="orthogonal", dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
            t2 = MLP(layers_list, init="orthogonal", dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
            flows += [MaskedCondAffineFlow(b, t1, s)]
            flows += [MaskedCondAffineFlow(1 - b, t2, s)]
            # flows += [LULinearPermute(action_sizes)]
        
        # Construct scaling network and preprocessing
        scale_list = [state_sizes] + [scale_hidden_sizes]*hidden_layers + [1]
        learnable_scale_1 = MLP(scale_list, init="zero", dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
        learnable_scale_2 = MLP(scale_list, init="zero", dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
        flows += [CondScaling(learnable_scale_1, learnable_scale_2)]
        flows += [Preprocessing(option='atanh', clip=True)]
        return flows, q0

def _train(cfg):
    # seed for reproducibility
    set_seed()  # e.g. `set_seed(42)` for fixed seed

    # load and wrap the Omniverse Isaac Gym environment
    # Ref: https://github.com/ray-project/ray/issues/3265#issuecomment-510215566
    # del os.environ['CUDA_VISIBLE_DEVICES']
    env = load_omniverse_isaacgym_env(
        task_name=cfg['task_name'],
        headless=True,
        num_envs=cfg['num_envs'],
        parse_args=False,
    )
    env = wrap_env(env)
    device = env.device
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=cfg["memory_size"], num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, alpha=cfg['entropy_value'], sigma_max=cfg['sigma_max'], sigma_min=cfg['sigma_min'])
    models["target_policy"] = Policy(env.observation_space, env.action_space, device, alpha=cfg['entropy_value'], sigma_max=cfg['sigma_max'], sigma_min=cfg['sigma_min'])

    agent = EBFlow(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": cfg["timesteps"], "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()

def _test(cfg):
    set_seed()

    path_load = cfg["experiment"]["directory"]
    cfg["experiment"]["directory"] = "runs/torch/"+cfg['task_name']

    env = load_omniverse_isaacgym_env(
        task_name=cfg['task_name'],
        headless=False,
        num_envs=1,
        parse_args=False,
    )
    env = wrap_env(env)
    device = env.device
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=cfg["memory_size"], num_envs=env.num_envs, device=device)

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device, alpha=cfg['entropy_value'], sigma_max=cfg['sigma_max'], sigma_min=cfg['sigma_min'])
    models["target_policy"] = Policy(env.observation_space, env.action_space, device, alpha=cfg['entropy_value'], sigma_max=cfg['sigma_max'], sigma_min=cfg['sigma_min'])

    agent = EBFlow(models=models,
                    memory=memory,
                    cfg=cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    
    agent.load(path_load)

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": cfg["timesteps"], "headless": False}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.eval()

def main():
    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = EBFlow_DEFAULT_CONFIG.copy()
    cfg["task_name"] = "Ingenuity"
    cfg["batch_size"] = 4096
    cfg["polyak"] = 0.005
    cfg["learning_rate"] = 5e-4
    cfg["grad_norm_clip"] = 10
    cfg["num_envs"] = 64
    cfg["timesteps"] = 160000
    cfg["experiment"]["directory"] = "runs/torch/AllegroHand"
    _train(cfg)

if __name__ == '__main__':
    main()