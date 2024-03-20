import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ebflow import EBFlow, EBFlow_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, FlowMixin
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

from normflows.nets import MLP
from normflows.transforms import Preprocessing
from normflows.distributions import MyConditionalDiagGaussian
from normflows.flows import MaskedCondAffineFlow, CondScaling, LULinearPermute

# define models (stochastic and deterministic models) using mixins
class Policy(FlowMixin, Model):
    def __init__(self, observation_space, action_space, device, alpha, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        FlowMixin.__init__(self, reduction)
        flows, q0 = self.init_Coupling(self.num_observations, self.num_actions)
        self.flows = nn.ModuleList(flows).to(self.device)
        self.prior = q0.to(self.device)
        self.alpha = alpha
        self.unit_test(num_samples=10, scale=1/0.5)

    def init_Coupling(self, state_sizes, action_sizes):
        init_zeros = True
        activation = 'swish'
        dropout_rate_flow = 0.1
        dropout_rate_scale = 0
        layer_norm_flow = True
        layer_norm_scale = False
        sigma_max = -0.3
        sigma_min = -5
        prior_transform = True # set True to perform additional tanh scaling for unlearnable prior.
        hidden_sizes = 64
        scale_hidden_sizes = 256
        hidden_layers = 2
        flow_layers = 2
        
        # Construct prior distribution
        prior_list = [state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
        loc = None
        log_scale = MLP(prior_list, init_zeros=init_zeros, activation=activation)
        q0 = MyConditionalDiagGaussian(action_sizes, loc, log_scale, SIGMA_MIN=sigma_min, SIGMA_MAX=sigma_max, prior_transform=prior_transform)

        # Construct flow network
        flows = []
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_sizes)])
        for i in range(flow_layers):
            layers_list = [action_sizes+state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
            s = None
            t1 = MLP(layers_list, init_zeros=init_zeros, activation=activation, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
            t2 = MLP(layers_list, init_zeros=init_zeros, activation=activation, dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
            flows += [MaskedCondAffineFlow(b, t1, s)]
            flows += [MaskedCondAffineFlow(1 - b, t2, s)]
            flows += [LULinearPermute(action_sizes)]
        
        # Construct scaling network and preprocessing
        scale_list = [state_sizes] + [scale_hidden_sizes]*hidden_layers + [1]
        learnable_scale_1 = MLP(scale_list, init_zeros=init_zeros, activation=activation, dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
        learnable_scale_2 = MLP(scale_list, init_zeros=init_zeros, activation=activation, dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
        flows += [CondScaling(learnable_scale_1, learnable_scale_2)]
        flows += [Preprocessing(option='scaleatanh', clip=True, scale=0.5)]
        return flows, q0


# load and wrap the gym environment.
# note: the environment version may change depending on the gym version
try:
    env = gym.make("Pendulum-v1")
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)
device = env.device

# instantiate a memory as experience replay
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = EBFlow_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1 #1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75
cfg["experiment"]["checkpoint_interval"] = 750
cfg["experiment"]["directory"] = "runs/torch/Pendulum"

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, alpha=cfg['entropy_value'])
models["target_policy"] = Policy(env.observation_space, env.action_space, device, alpha=cfg['entropy_value'])

agent = EBFlow(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
