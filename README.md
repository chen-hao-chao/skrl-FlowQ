# Experiments on the Omniverse Isaac Gym Environments (SKRL Implementation)

**skrl** is an open-source modular library for Reinforcement Learning written in Python (on top of [PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io)) and designed with a focus on modularity, readability, simplicity, and transparency of algorithm implementation. In addition to supporting the OpenAI [Gym](https://www.gymlibrary.dev) / Farama [Gymnasium](https://gymnasium.farama.org) and [DeepMind](https://github.com/deepmind/dm_env) and other environment interfaces, it allows loading and configuring [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/), [NVIDIA Isaac Orbit](https://isaac-orbit.github.io/orbit/index.html) and [NVIDIA Omniverse Isaac Gym](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_gym_isaac_gym.html) environments, enabling agents' simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run.

<br>

### Please, visit the documentation for usage details and examples

<strong>https://skrl.readthedocs.io</strong>

<br>

> **Note:** This project is under **active continuous development**. Please make sure you always have the latest version. Visit the [develop](https://github.com/Toni-SM/skrl/tree/develop) branch or its [documentation](https://skrl.readthedocs.io/en/develop) to access the latest updates to be released.

<br>

### Reproducing the Results of SAC
```
PYTHON_PATH tune_sac_1.py
PYTHON_PATH tune_sac_2.py
PYTHON_PATH tune_sac_3.py
PYTHON_PATH tune_sac_4.py
PYTHON_PATH tune_sac_5.py
PYTHON_PATH tune_sac_6.py
```

### Reproducing the Results of MEow
```
PYTHON_PATH tune_ebflow_1.py
PYTHON_PATH tune_ebflow_2.py
PYTHON_PATH tune_ebflow_3.py
PYTHON_PATH tune_ebflow_4.py
PYTHON_PATH tune_ebflow_5.py
PYTHON_PATH tune_ebflow_6.py
```

### Citing SKRL library
To cite the library in publications, please use the following reference:

```bibtex
@article{serrano2023skrl,
  author  = {Antonio Serrano-Muñoz and Dimitrios Chrysostomou and Simon Bøgh and Nestor Arana-Arexolaleiba},
  title   = {skrl: Modular and Flexible Library for Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {254},
  pages   = {1--9},
  url     = {http://jmlr.org/papers/v24/23-0112.html}
}
```