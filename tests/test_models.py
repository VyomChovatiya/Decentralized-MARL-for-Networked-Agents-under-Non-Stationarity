"""
Tests for Neural Architectures and MARL Trainers (Track 2)
"""
import numpy as np
import torch
from src.models.actors import DecentralizedMLPActor
from src.models.critics import DecentralizedMLPCritic
from src.models.gnn_comm import GNNCommunicationLayer
from src.models.bayes_ego import BayesGEgoInference
from src.algorithms.ippo import IPPOTrainer
from src.algorithms.networked_ppo import NetworkedPPOTrainer
from src.algorithms.bayesg import BayesGTrainer


def test_actor_critic_shapes():
    obs_dim = 3
    act_dim = 2
    batch_size = 4

    actor = DecentralizedMLPActor(obs_dim, act_dim)
    critic = DecentralizedMLPCritic(obs_dim)

    obs = torch.randn(batch_size, obs_dim)
    action, log_prob = actor.get_action(obs)
    val = critic(obs)

    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert val.shape == (batch_size,)


def test_gnn_message_passing():
    num_agents = 5
    feat_dim = 8
    out_dim = 16

    gnn = GNNCommunicationLayer(feat_dim, out_dim, k_hops=2)
    features = torch.randn(num_agents, feat_dim)
    adj = torch.eye(num_agents)

    out = gnn(features, adj)
    assert out.shape == (num_agents, out_dim)


def test_bayes_ego_inference():
    num_agents = 5
    feat_dim = 4

    bayes = BayesGEgoInference(feat_dim=feat_dim)
    features = torch.randn(num_agents, feat_dim)
    adj = torch.ones(num_agents, num_agents)

    mask, kl_loss = bayes(features, adj)
    assert mask.shape == (num_agents, num_agents)
    assert kl_loss.item() >= 0.0


def test_all_three_trainers_update():
    num_agents = 5
    obs_dim = 3
    act_dim = 2
    adj = torch.ones(num_agents, num_agents)

    for trainer in [
        IPPOTrainer(obs_dim=obs_dim, act_dim=act_dim, epochs=2),
        NetworkedPPOTrainer(obs_dim=obs_dim, act_dim=act_dim, k_hops=1, gnn_dim=16, epochs=2),
        BayesGTrainer(obs_dim=obs_dim, act_dim=act_dim, k_hops=1, gnn_dim=16, epochs=2),
    ]:
        for t in range(4):
            obs = torch.randn(num_agents, obs_dim)
            out = trainer.select_action(obs, adj)
            acts, logp, vals = out[0], out[1], out[2]
            rews = np.ones(num_agents, dtype=np.float32)
            trainer.store_transition(obs, acts, logp, rews, t == 3, vals, adj=adj)

        stats = trainer.update(adj) if hasattr(trainer, "k_hops") else trainer.update()
        assert "actor_loss" in stats
        assert "critic_loss" in stats
