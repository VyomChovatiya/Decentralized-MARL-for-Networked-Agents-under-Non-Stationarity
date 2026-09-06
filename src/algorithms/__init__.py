"""
Reinforcement Learning Algorithms Module (Track 2)
"""
from .ippo import IPPOTrainer
from .networked_ppo import NetworkedPPOTrainer
from .bayesg import BayesGTrainer

__all__ = [
    "IPPOTrainer",
    "NetworkedPPOTrainer",
    "BayesGTrainer",
]
