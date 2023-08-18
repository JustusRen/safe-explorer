# Implementation of "Safe Exploration in Continuous Action Spaces" including a guidance component

## Introduction

This repository contains Pytorch implementation of paper ["Safe Exploration in Continuous Action Spaces" (Dalal et al., 2018)](https://arxiv.org/pdf/1801.08757.pdf). This repository is a fork of https://github.com/kollerlukas/safe-explorer and extends it by adding a temporal logic-based guidance component.

## Setup

The code was tested with Python 3.7. 
```sh
pip install -r requirements.txt
```

To generate a DFA using the script generate_dfa.py or using the function provided in the Ball1D environment, the installation instructions of the following repository have to be used: https://github.com/whitemech/LTLf2DFA.

## Training

A list of parameters and their default values is printed with the following command.
```sh
python -m safe_explorer.main --help
```

With the following command the agent is trained on the ball domain.
```sh
python -m safe_explorer.main --main_trainer_task ballnd
```

The training can be monitored via Tensorboard with the following command.
```sh
tensorboard --logdir=runs
```

## References
- Dalal, G., K. Dvijotham, M. Vecerik, T. Hester, C. Paduraru, and Y. Tassa (2018). “Safe Exploration in Continuous Action Spaces.” In: CoRR abs/1801.08757. arXiv: 1801.08757.

- Lillicrap, T. P., J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra (May 2016). “Continuous control with deep reinforcement learning.” In: 4th International Conference on Learning Representations, (ICLR 2016), Conference Track Proceedings. Ed. by Y. Bengio and Y. LeCun. ICLR’16. San Juan, Puerto Rico.

## Acknowledgements

This repository was originally a fork from https://github.com/kollerlukas/safe-explorer. 

The *Deep Determinitic Policy Gradient* (DDPG) (Lillicrap et al., 2016) implementation is based on this implementation: [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b).
