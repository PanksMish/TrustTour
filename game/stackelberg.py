"""
Strategic Optimization Module (Section 3.7 / 4.3, Eq. 32-38, 52-59).

Models tourism verification as a leader-follower Stackelberg game:

    G = (P, A, S, U_P, U_A)                                        Eq. (32)/(52)

Platform strategy:   S_P = (theta_v, theta_r, theta_s)              Eq. (53)
Adversary strategy:  S_A = (m, q, eps)                              Eq. (54)
Platform utility:    U_P = l1*T - l2*R + l3*B - l4*C_H               Eq. (55)
Booking confidence:  B = (1/N) sum T(I_i)                            Eq. (56)
Adversary utility:   U_A = mu1*P_S - mu2*P_D - mu3*C_A                Eq. (57)
Equilibrium:         S_P* = argmax U_P(S_P, S_A*(S_P))                Eq. (58)
                      S_A*(S_P) = argmax U_A(S_P, S_A)                 Eq. (59)

We solve this via alternating best-response (fictitious-play style) gradient
ascent on a smooth, bounded parametrization of both strategy spaces, which
converges to a local Stackelberg/Nash-style equilibrium for the continuous,
compact strategy spaces assumed in Theorem 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GameConfig:
    iterations: int = 60
    lr_platform: float = 0.05
    lr_adversary: float = 0.05
    # platform utility weights, Eq. (55)
    lambda1: float = 1.0   # weight on average trust T
    lambda2: float = 0.7   # weight on average risk R
    lambda3: float = 0.8   # weight on booking confidence B
    lambda4: float = 0.3   # weight on human verification cost C_H
    # adversary utility weights, Eq. (57)
    mu1: float = 1.0       # weight on successful misinformation P_S
    mu2: float = 1.0       # weight on detection probability P_D
    mu3: float = 0.4       # weight on adversarial generation cost C_A
    seed: int = 7


class StackelbergGame:
    """Leader (platform) / follower (adversary) strategic optimization.

    Platform strategy S_P = (theta_v, theta_r, theta_s) in [0,1]^3:
        theta_v: verification intensity (higher -> more scrutiny, higher C_H, higher P_D)
        theta_r: recommendation priority for verified content (affects B)
        theta_s: safety-alert sensitivity (affects R, C_H)

    Adversary strategy S_A = (m, q, eps) in [0,1]^3:
        m:   manipulation intensity
        q:   visual quality of generated content (higher -> harder to detect)
        eps: adversarial perturbation strength (evasion effort, also raises C_A)
    """

    def __init__(self, cfg: GameConfig | None = None):
        self.cfg = cfg or GameConfig()
        rng = np.random.default_rng(self.cfg.seed)
        self.s_p = rng.uniform(0.4, 0.6, size=3)  # theta_v, theta_r, theta_s
        self.s_a = rng.uniform(0.4, 0.6, size=3)  # m, q, eps
        self.history = {"U_P": [], "U_A": [], "s_p": [], "s_a": []}

    # ---- core probabilistic mappings tying strategies to T, R, B, P_S, P_D ----

    def _detection_prob(self, s_p: np.ndarray, s_a: np.ndarray) -> float:
        theta_v = s_p[0]
        q, eps = s_a[1], s_a[2]
        # Higher verification intensity raises detection; higher generated
        # quality / evasion perturbation lowers it.
        z = 3.0 * theta_v - 2.5 * q - 1.5 * eps
        return float(_sigmoid(z))

    def _avg_trust(self, s_p: np.ndarray, s_a: np.ndarray) -> float:
        theta_v, theta_s = s_p[0], s_p[2]
        m, q = s_a[0], s_a[1]
        z = 2.0 * theta_v + 1.0 * theta_s - 2.5 * m - 1.0 * q + 1.0
        return float(_sigmoid(z))

    def _avg_risk(self, avg_trust: float) -> float:
        return 1.0 - avg_trust  # Eq. (46)/(16) with omega=1

    def _booking_confidence(self, s_p: np.ndarray, avg_trust: float) -> float:
        theta_r = s_p[1]
        return float(np.clip(avg_trust * (0.5 + 0.5 * theta_r), 0.0, 1.0))  # Eq. (56)

    def _human_cost(self, s_p: np.ndarray) -> float:
        theta_v, theta_s = s_p[0], s_p[2]
        return float(np.clip(0.6 * theta_v + 0.4 * theta_s, 0.0, 1.0))

    def _success_prob(self, detection_prob: float) -> float:
        return float(np.clip(1.0 - detection_prob, 0.0, 1.0))  # P_S

    def _adversary_cost(self, s_a: np.ndarray) -> float:
        q, eps = s_a[1], s_a[2]
        return float(np.clip(0.5 * q + 0.5 * eps, 0.0, 1.0))  # C_A

    # ---- utilities, Eq. (55) / (57) ----

    def platform_utility(self, s_p: np.ndarray, s_a: np.ndarray) -> float:
        cfg = self.cfg
        avg_trust = self._avg_trust(s_p, s_a)
        avg_risk = self._avg_risk(avg_trust)
        booking = self._booking_confidence(s_p, avg_trust)
        ch = self._human_cost(s_p)
        return cfg.lambda1 * avg_trust - cfg.lambda2 * avg_risk + cfg.lambda3 * booking - cfg.lambda4 * ch

    def adversary_utility(self, s_p: np.ndarray, s_a: np.ndarray) -> float:
        cfg = self.cfg
        pd = self._detection_prob(s_p, s_a)
        ps = self._success_prob(pd)
        ca = self._adversary_cost(s_a)
        return cfg.mu1 * ps - cfg.mu2 * pd - cfg.mu3 * ca

    # ---- numerical best-response gradients (finite differences) ----

    @staticmethod
    def _finite_diff_grad(f, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] = np.clip(xp[i] + eps, 0, 1)
            xm = x.copy(); xm[i] = np.clip(xm[i] - eps, 0, 1)
            grad[i] = (f(xp) - f(xm)) / (2 * eps)
        return grad

    def step(self):
        cfg = self.cfg
        # Follower best-response: adversary maximizes U_A given current platform strategy
        grad_a = self._finite_diff_grad(lambda sa: self.adversary_utility(self.s_p, sa), self.s_a)
        self.s_a = np.clip(self.s_a + cfg.lr_adversary * grad_a, 0.0, 1.0)

        # Leader update: platform maximizes U_P anticipating adversary's response
        grad_p = self._finite_diff_grad(lambda sp: self.platform_utility(sp, self.s_a), self.s_p)
        self.s_p = np.clip(self.s_p + cfg.lr_platform * grad_p, 0.0, 1.0)

        up = self.platform_utility(self.s_p, self.s_a)
        ua = self.adversary_utility(self.s_p, self.s_a)
        self.history["U_P"].append(up)
        self.history["U_A"].append(ua)
        self.history["s_p"].append(self.s_p.copy())
        self.history["s_a"].append(self.s_a.copy())
        return up, ua

    def solve(self, iterations: int | None = None, verbose: bool = True) -> dict:
        iterations = iterations or self.cfg.iterations
        for it in range(iterations):
            up, ua = self.step()
            if verbose and (it + 1) % max(1, iterations // 10) == 0:
                print(f"[Stackelberg] iter {it+1}/{iterations}  U_P={up:.4f}  U_A={ua:.4f}  "
                      f"S_P={np.round(self.s_p, 3)}  S_A={np.round(self.s_a, 3)}")
        return {
            "platform_strategy": {"theta_v": self.s_p[0], "theta_r": self.s_p[1], "theta_s": self.s_p[2]},
            "adversary_strategy": {"m": self.s_a[0], "q": self.s_a[1], "eps": self.s_a[2]},
            "U_P": self.history["U_P"][-1],
            "U_A": self.history["U_A"][-1],
            "history": self.history,
        }
