# Upward-Driven Disk

This project is my attempt to reproduce the mathematical model and results from Leo Maas's 2025 paper, *The Upward-Driven Disk: A Steadily Forced Chaotic Pendulum*.

Paper: [Leo Maas (2025), American Journal of Physics](https://arxiv.org/abs/2505.17957)

The setup is relatively simple: a disk is pinched between two rollers, with one of the rollers driving it upwards. Depending on the driving speed and initial conditions, the disk can exhibit chaotic, periodic, or steady rotational motion.

What makes this system interesting is that a relatively simple mechanical setup can produce surprisingly complicated nonlinear dynamics. My goal is to first reproduce the mathematical model and results from the paper, understand the underlying mechanics, and eventually investigate some of the behaviour that the original model does not fully explain.

This repository contains the numerical implementation, figure reproduction scripts, and additional analysis.

---

## 1. The full mathematical model

The main model is based on the equations derived in Maas's paper.

The system is described by three state variables:

- $Y$: horizontal position of the disk's centre relative to the pivot.
- $Z$: vertical position of the disk's centre relative to the pivot.
- $\Omega$: angular velocity of the disk, taken as positive clockwise.

The driving speed of the roller is represented by $W$, while $a$ and $g$ denote the disk radius and gravitational acceleration respectively.

The equations of motion are

$$
\dot{Y}=Z\Omega
$$

$$
\dot{Z}=-Y\Omega+W
$$

$$
\mathcal{I}\dot{\Omega}
=
gY-k(\Omega)\Omega-WZ\Omega
$$

where the moment of inertia about the instantaneous pivot is

$$
\mathcal{I}=\frac{1}{2}a^2+Y^2+Z^2
$$

The damping coefficient depends on the angular velocity:

$$
k(\Omega)=\max\left(k_1-k_2|\Omega|,\ k_0\right)
$$

The angular momentum-like quantity defined in the paper is

$$
\mathcal{J}=\mathcal{I}\Omega-WY
$$

This is useful when analysing the system's dynamics and checking the numerical implementation.

### Parameters

The parameters used in the simulations are:

| Parameter | Value | Unit |
|---|---:|---|
| $k_0$ | 0.012 | $\mathrm{m^2/s}$ |
| $k_1$ | 0.103 | $\mathrm{m^2/s}$ |
| $k_2$ | 0.055 | $\mathrm{m^2}$ |
| $a$ | 0.08 | $\mathrm{m}$ |
| $g$ | 9.81 | $\mathrm{m/s^2}$ |

The damping parameters and disk radius are taken from the paper. The value of $g$ is set to the standard gravitational acceleration, as the paper does not explicitly specify its numerical value.

---

## 2. The dimensionless model (DLE)

Maas also derives a dimensionless version of the equations, which I refer to as the dimensionless equations (DLE).

The dimensionless state variables are $y$, $z$, and $\omega$, while $w$ represents the dimensionless driving speed.

The resulting equations are

$$
\dot{\omega}=y-\omega
$$

$$
\dot{y}=\omega z
$$

$$
\dot{z}=-\omega y+w
$$

The DLE is considerably simpler than the full dimensional model, making it useful for analysing the system's nonlinear dynamics.

However, it is important to distinguish between the two models. The DLE is a reduced dimensionless formulation and should not automatically be treated as equivalent to the full model in every regime.

The characteristic length and time scales are

$$
L=\frac{2k_0^2}{ga^2}
$$

$$
\tau=\frac{a^2}{2k_0}
$$

The dimensional and dimensionless variables are related by

$$
Y=Ly,\qquad Z=Lz
$$

$$
\Omega=\frac{\omega}{\tau}
$$

and

$$
w=\frac{\tau W}{L}
$$

The dimensionless time is defined by

$$
t=\frac{t_{\mathrm{dim}}}{\tau}
$$

where $t_{\mathrm{dim}}$ denotes physical time. Consequently, differentiation with respect to dimensionless time is different from differentiation with respect to physical time.

---

## 3. The modified DLE

One limitation of the original DLE is that it does not reproduce the stable steady-rotation states observed experimentally.

Maas discusses a modified version of the dimensionless equations by introducing an additional drive-dependent damping term.

The modified equations are

$$
\dot{\omega}=y-\omega
$$

$$
\dot{y}=\omega z
$$

$$
\dot{z}=-\omega y+w-Bwz
$$

where $B$ controls the strength of the additional damping.

This modification allows stable rotating equilibria to exist under suitable conditions.

I implemented this model separately from the original DLE so that the original equations remain unchanged and the effects of the additional damping can be studied independently.

---

## 4. Numerical implementation

The simulations use Python, NumPy, SciPy, and Matplotlib.

Numerical integration is performed using SciPy's `solve_ivp`, with the DOP853 method.

### Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Make the figures

This should take roughly 1 minute, and the figures should all be inside the "figures" folder

```bash
python3 reproduce.py
```

### To make simulation modifications

As this simulation was made with the idea of being a master folder, to make a new simulation with differing variable values, simply make a new python file and import the maas_model and do the adjustments.

For example:
```bash
from maas_model import run_full, run_dle, lyapunov_full, P0

# full model: start (Y, Z, Omega) in SI, drive W in m/s, duration in s
t, Y, Z, Om, J, lost = run_full([0.005, 0.04, 0.0], 0.039, 60.0)

# change the apparatus: a 6 cm disc, stiffer friction
p = P0.with_(a=0.06, k0=0.015)
t, Y, Z, Om, J, lost = run_full([0.005, 0.03, 0.0], 0.039, 60.0, p=p)
print(p.w_of_W(0.039))                       # the dimensionless drive this corresponds to

t, om, y, z = run_dle([0.0, 1.0, -2.0], 3.9, 400.0)          # DLE, dimensionless
lam = lyapunov_full([0.005, 0.04, 0.0], 0.039)               # > 0 means chaotic
```