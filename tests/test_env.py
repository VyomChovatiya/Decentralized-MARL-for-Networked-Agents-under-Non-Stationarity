"""
Tests for Networked Consensus Environment (Track 1)
"""
from src.environments.toy_consensus import NetworkedConsensusEnv


def test_env_reset_and_step():
    env = NetworkedConsensusEnv(num_agents=5, topology_type="ring", max_steps=10)
    obs, info = env.reset(seed=42)

    assert len(obs) == 5
    for agent_id, agent_obs in obs.items():
        assert agent_obs.shape == (3,)

    # Step with no flips
    actions = {agent_id: 0 for agent_id in env.agents}
    next_obs, rewards, terms, truncs, step_info = env.step(actions)

    assert len(next_obs) == 5
    assert len(rewards) == 5
    assert isinstance(step_info["consensus"], bool)
