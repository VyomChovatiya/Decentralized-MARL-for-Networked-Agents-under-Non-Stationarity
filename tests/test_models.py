"""
Tests for Neural Architectures (Track 2)
"""
import torch
from src.models.actors import DecentralizedMLPActor
from src.models.critics import DecentralizedMLPCritic
from src.models.gnn_comm import GNNCommunicationLayer
from src.models.bayes_ego import BayesGEgoInference


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
