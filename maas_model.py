"""maas_model.py -- Maas (2025), "The upward-driven disk", in the paper's own conventions.

Full model (Sec. IV, per unit mass; state (Y, Z, Omega), Omega positive CLOCKWISE):
    Y' = Z Omega                                   (8a)
    Z' = -Y Omega + W                              (8b)
    I Omega' = g Y - k(Omega) Omega - W Z Omega    (11),  I = a^2/2 + Y^2 + Z^2  (4-5)
    k(Omega) = max(k1 - k2 |Omega|, k0)            (18)
    J = I Omega - W Y                              (9)   angular momentum per unit mass about the pivot

Diffusionless Lorenz equations (Sec. IV C; state (omega, y, z)):
    omega' = y - omega,  y' = omega z,  z' = -omega y + w                      (14)
    scales  L = 2 k0^2/(g a^2),  tau = a^2/(2 k0),  w = tau W / L             (13)
    rotating equilibria  omega = y = +-sqrt(w), z = 0                          (19)
    characteristic polynomial  lambda^3 + lambda^2 + w lambda + 2 w = 0        (20)
    damped variant  z' = -omega y + w - B w z                                  (21)

Additions that are NOT in the paper but are needed to run it honestly:
    * a terminal event when the pivot reaches the rim (sqrt(Y^2+Z^2) = a): the equations
      describe nothing after that, the disc has left the wheels;
    * g = 9.81 m/s^2 as an input (the paper prints only the rounded scales L = 4.6 mm,
      tau = 0.27 s, which are what g = 9.81 gives);
    * a largest-Lyapunov-exponent routine (tangent-space / Benettin) to classify regimes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class MaasParams:
    a: float = 0.08        # disk radius [m]        (Sec. II A)
    g: float = 9.81        # gravity [m/s^2]
    k0: float = 0.012      # [m^2/s]  (18)
    k1: float = 0.103      # [m^2/s]  (18)
    k2: float = 0.055      # [m^2]    (18)

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def L(self) -> float:          # (13)
        return 2.0 * self.k0 ** 2 / (self.g * self.a ** 2)

    @property
    def tau(self) -> float:        # (13)
        return self.a ** 2 / (2.0 * self.k0)

    def w_of_W(self, W: float) -> float:   # (13)
        return self.tau * W / self.L

    def W_of_w(self, w: float) -> float:
        return w * self.L / self.tau


P0 = MaasParams()


# ---------------------------------------------------------------- full model
def k_of(Omega, p: MaasParams = P0):
    return np.maximum(p.k1 - p.k2 * np.abs(Omega), p.k0)


def inertia(Y, Z, p: MaasParams = P0):
    return 0.5 * p.a ** 2 + Y * Y + Z * Z


def angular_momentum(Y, Z, Omega, W, p: MaasParams = P0):
    return inertia(Y, Z, p) * Omega - W * Y


def full_rhs(t, s, W, p: MaasParams = P0):
    Y, Z, Om = s
    I = inertia(Y, Z, p)
    k = p.k1 - p.k2 * abs(Om)
    if k < p.k0:
        k = p.k0
    return [Z * Om, -Y * Om + W, (p.g * Y - k * Om - W * Z * Om) / I]


def full_jac(s, W, p: MaasParams = P0):
    """Jacobian of full_rhs (for the Lyapunov exponent)."""
    Y, Z, Om = s
    I = inertia(Y, Z, p)
    k = p.k1 - p.k2 * abs(Om)
    dk = -p.k2 * np.sign(Om)
    if k < p.k0:
        k, dk = p.k0, 0.0
    N = p.g * Y - k * Om - W * Z * Om
    J = np.zeros((3, 3))
    J[0] = [0.0, Om, Z]
    J[1] = [-Om, 0.0, -Y]
    J[2, 0] = p.g / I - N * 2 * Y / I ** 2
    J[2, 1] = -W * Om / I - N * 2 * Z / I ** 2
    J[2, 2] = (-k - dk * Om - W * Z) / I
    return J


def make_lost_event(p: MaasParams = P0):
    def lost(t, s, *args):
        return math.hypot(s[0], s[1]) - p.a
    lost.terminal = True
    lost.direction = 1
    return lost


def run_full(s0, W, t_end, p: MaasParams = P0, dt=1e-3, rtol=1e-9, atol=1e-12):
    """Integrate (8)+(11). Returns t, Y, Z, Omega, J, lost(bool)."""
    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
    sol = solve_ivp(full_rhs, (0.0, t_end), np.asarray(s0, float), args=(W, p), t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol, events=make_lost_event(p))
    Y, Z, Om = sol.y
    return sol.t, Y, Z, Om, angular_momentum(Y, Z, Om, W, p), sol.status == 1


def lyapunov_full(s0, W, p: MaasParams = P0, t_skip=30.0, t_total=150.0, dt=0.5, rtol=1e-9, atol=1e-12):
    """Largest Lyapunov exponent [1/s] of the full model (NaN if the disc is lost)."""
    lost = make_lost_event(p)

    def aug(t, u):
        x, d = u[:3], u[3:]
        return np.concatenate([full_rhs(t, x, W, p), full_jac(x, W, p) @ d])

    x = np.asarray(s0, float)
    sol = solve_ivp(full_rhs, (0, t_skip), x, args=(W, p), method="DOP853", rtol=rtol, atol=atol, events=lost)
    if sol.status == 1:
        return float("nan")
    x = sol.y[:, -1]
    d = np.array([1.0, 0.0, 0.0])
    acc = 0.0
    n = int(round(t_total / dt))
    for _ in range(n):
        sol = solve_ivp(aug, (0, dt), np.concatenate([x, d]), method="DOP853", rtol=rtol, atol=atol,
                        events=lambda t, u: lost(t, u[:3]))
        if sol.status == 1:
            return float("nan")
        x, d = sol.y[:3, -1], sol.y[3:, -1]
        nd = np.linalg.norm(d)
        acc += math.log(nd)
        d /= nd
    return acc / (n * dt)


# ---------------------------------------------------------------- DLE
def dle_rhs(t, s, w, B=0.0):
    """(14), or (21) when B != 0.  State (omega, y, z)."""
    om, y, z = s
    return [y - om, om * z, -om * y + w - B * w * z]


def run_dle(s0, w, t_end, B=0.0, dt=1e-2, rtol=1e-10, atol=1e-12):
    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
    sol = solve_ivp(dle_rhs, (0.0, t_end), np.asarray(s0, float), args=(w, B), t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def dle_equilibria(w):
    """(19): (omega, y, z) = (+-sqrt(w), +-sqrt(w), 0)."""
    r = math.sqrt(w)
    return [np.array([r, r, 0.0]), np.array([-r, -r, 0.0])]


def dle_jacobian(s, w, B=0.0):
    om, y, z = s
    return np.array([[-1.0, 1.0, 0.0],
                     [z, 0.0, om],
                     [-y, -om, -B * w]])


def dle_eigenvalues(w, B=0.0):
    """Eigenvalues at the rotating equilibrium (+sqrt(w), +sqrt(w), 0).  With B = 0 they are the
    roots of (20): lambda^3 + lambda^2 + w lambda + 2 w = 0."""
    return np.linalg.eigvals(dle_jacobian(dle_equilibria(w)[0], w, B))


def charpoly_roots(w):
    return np.roots([1.0, 1.0, w, 2.0 * w])
