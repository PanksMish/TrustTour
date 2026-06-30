import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.ds_svit import build_ds_svit
from models.trust_module import TrustAssessmentModule, TrustModuleConfig
from models.decision_module import AdaptiveDecisionModule, DecisionThresholds, Decision
from rl.hitl_rl import HITLRLAgent, HITLRLConfig
from game.stackelberg import StackelbergGame, GameConfig


def _build_trust_module():
    backbone = build_ds_svit(img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=4, fusion_heads=4)
    cfg = TrustModuleConfig(feature_dim=64, metadata_dim=8, source_dim=6, hidden_dim=32)
    return TrustAssessmentModule(backbone, cfg)


def test_trust_module_forward():
    model = _build_trust_module()
    image = torch.randn(4, 3, 64, 64)
    metadata = torch.rand(4, 8)
    source = torch.zeros(4, 6)
    source[:, 0] = 1.0

    out = model(image, metadata, source)
    assert out["trust"].shape == (4,)
    assert torch.all((out["trust"] >= 0) & (out["trust"] <= 1))
    # weights should sum to 1 (softmax)
    assert torch.allclose(out["weights"].sum(dim=-1), torch.ones(4), atol=1e-5)


def test_decision_module_thresholds():
    dm = AdaptiveDecisionModule(DecisionThresholds(0.3, 0.7))
    trust = torch.tensor([0.95, 0.5, 0.05])  # risk = [0.05, 0.5, 0.95]
    decisions = dm.decide(trust)
    assert decisions[0].item() == Decision.VERIFIED
    assert decisions[1].item() == Decision.HUMAN_REVIEW
    assert decisions[2].item() == Decision.REJECTED


def test_decision_module_risk_monotonic():
    dm = AdaptiveDecisionModule(DecisionThresholds(0.3, 0.7), impact_factor=1.0)
    trust_high = torch.tensor([0.9])
    trust_low = torch.tensor([0.1])
    assert dm.risk(trust_high).item() < dm.risk(trust_low).item()


def test_hitl_rl_runs_and_updates_thresholds():
    torch.manual_seed(0)
    trust = torch.rand(200)
    auth = torch.rand(200)
    expert = (torch.rand(200) > 0.4).float()

    agent = HITLRLAgent(HITLRLConfig(episodes=20, batch_size=64, seed=1))
    result = agent.run(trust, auth, expert, verbose=False)
    assert 0.0 < result["tau1"] < result["tau2"] < 1.0
    assert len(result["reward_history"]) == 20
    assert 0.0 <= result["human_review_rate"] <= 1.0


def test_stackelberg_game_converges_to_compact_strategies():
    game = StackelbergGame(GameConfig(iterations=15, seed=3))
    result = game.solve(verbose=False)
    sp = result["platform_strategy"]
    sa = result["adversary_strategy"]
    for v in list(sp.values()) + list(sa.values()):
        assert 0.0 <= v <= 1.0
    assert isinstance(result["U_P"], float)
    assert isinstance(result["U_A"], float)


if __name__ == "__main__":
    test_trust_module_forward()
    test_decision_module_thresholds()
    test_decision_module_risk_monotonic()
    test_hitl_rl_runs_and_updates_thresholds()
    test_stackelberg_game_converges_to_compact_strategies()
    print("All trust/decision/RL/game tests passed.")
