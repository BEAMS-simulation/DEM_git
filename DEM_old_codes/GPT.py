import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------
E_eff = 24e9          # Pa
R_eff = 2.5e-6        # m
alpha = 0.65          # real/smooth contact area ratio
sigma_H = 1.9e9       # Pa
tau_rel = 2.0e-4      # s, plastic relaxation time (tuning parameter)

# Optional simple cohesion model for negative-force tail during unloading
use_cohesion = True
k_coh = 8.0           # dimensionless multiplier for cohesive spring
F_tensile_max = 40e-6 # N, tensile cutoff (roughly to mimic figure)

# Numerical settings
n_steps = 2000
t_total = 1.0
dt = t_total / n_steps

# Fig. 2(c)-like max forces [N]
Fmax_list = [50e-6, 100e-6, 200e-6]

# -----------------------------
# Model functions
# -----------------------------
def k_spring(h):
    """Hertz-type spring coefficient."""
    h_eff = max(h, 0.0)
    return (4.0 / 3.0) * E_eff * np.sqrt(R_eff * h_eff)

def F_compressive_from_h(h, h_eq):
    """Compressing contact force from overlap and equilibrium overlap."""
    if h <= 0:
        return 0.0
    return k_spring(h) * max(h - h_eq, 0.0)

def contact_area(h):
    """A_con = pi * alpha * R_eff * h."""
    return np.pi * alpha * R_eff * max(h, 0.0)

def F_threshold(h):
    """F_th = (2/3) * sigma_H * A_con."""
    return (2.0 / 3.0) * sigma_H * contact_area(h)

def F_tensile_from_h(h, h_eq):
    """
    Simple cohesive branch:
    when unloading below h_eq, allow a tensile force until cutoff.
    This is not the exact DEM cohesion law from the paper,
    only a qualitative add-on to mimic Fig. 2(c).
    """
    if not use_cohesion:
        return 0.0
    if h >= h_eq:
        return 0.0
    # cohesive stiffness scaled by local Hertz stiffness at h_eq
    k_ref = k_spring(max(h_eq, 1e-12))
    Ft = -k_coh * k_ref * (h_eq - h)
    return max(Ft, -F_tensile_max)

def F_total_from_h(h, h_eq):
    """
    Total force branch:
    compressive if h >= h_eq,
    cohesive tensile if h < h_eq.
    """
    if h >= h_eq:
        return F_compressive_from_h(h, h_eq)
    return F_tensile_from_h(h, h_eq)

def solve_h_for_force(F_target, h_eq):
    """
    Force-controlled quasistatic solve:
    find h such that F_total_from_h(h, h_eq) = F_target.
    Since loading in Fig.2(c) is imposed force, we solve h numerically.

    For F_target >= 0, solve on compressive branch.
    For F_target < 0, solve on tensile branch.
    """
    if F_target >= 0:
        # bracket on compressive branch: h >= h_eq
        lo = max(h_eq, 0.0)
        hi = max(lo + 1e-12, 1e-9)
        while F_total_from_h(hi, h_eq) < F_target:
            hi *= 1.5
            if hi > 1e-3:
                raise RuntimeError("Failed to bracket compressive root.")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if F_total_from_h(mid, h_eq) < F_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # tensile branch
    if not use_cohesion:
        return max(h_eq, 0.0)

    hi = max(h_eq, 0.0)
    lo = max(0.0, hi - 1e-9)

    # extend lo leftward until enough tensile force is bracketed
    while F_total_from_h(lo, h_eq) > F_target:
        span = hi - lo
        lo = max(0.0, lo - max(1e-12, 1.5 * span))
        if lo <= 0.0 and F_total_from_h(lo, h_eq) > F_target:
            # if tensile target beyond cutoff, just saturate
            return 0.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if F_total_from_h(mid, h_eq) > F_target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def triangular_force_history(Fmax, n_steps):
    """
    Force increases linearly to Fmax, then decreases linearly.
    If cohesion is enabled, continue slightly into negative force
    to show the tensile tail qualitatively.
    """
    half = n_steps // 2
    up = np.linspace(0.0, Fmax, half, endpoint=False)
    if use_cohesion:
        down = np.linspace(Fmax, -0.6 * F_tensile_max, n_steps - half)
    else:
        down = np.linspace(Fmax, 0.0, n_steps - half)
    return np.concatenate([up, down])

def run_simulation(Fmax):
    F_hist = triangular_force_history(Fmax, n_steps)

    h_eq = 0.0
    h_list = []
    F_list = []
    h_eq_list = []
    Fth_list = []

    for F_app in F_hist:
        # 1) quasistatic overlap solving under imposed force
        h = solve_h_for_force(F_app, h_eq)

        # 2) plastic evolution only under compressive loading above threshold
        if h > 0:
            F_now = F_compressive_from_h(h, h_eq)
            Fth = F_threshold(h)
            ks = max(k_spring(h), 1e-30)

            if F_now > Fth:
                dh_eq = ((F_now - Fth) / (tau_rel * ks)) * dt
                # do not let h_eq exceed current overlap
                h_eq = min(h_eq + max(dh_eq, 0.0), h)
            Fth_list.append(Fth)
        else:
            Fth_list.append(0.0)

        h_list.append(h)
        F_list.append(F_app)
        h_eq_list.append(h_eq)

    return np.array(h_list), np.array(F_list), np.array(h_eq_list), np.array(Fth_list)


# -----------------------------
# Run and plot
# -----------------------------
plt.figure(figsize=(7, 5))

colors = ["tab:blue", "tab:orange", "goldenrod"]
labels = [
    r"$F_{\max}=50\ \mu N$",
    r"$F_{\max}=100\ \mu N$",
    r"$F_{\max}=200\ \mu N$",
]

# Hertzian reference (h_eq = 0)
h_ref = np.linspace(0, 120e-9, 400)
F_ref = np.array([F_compressive_from_h(h, 0.0) for h in h_ref])
plt.plot(h_ref * 1e9, F_ref * 1e6, "k--", lw=1.5, label="Hertzian contact law")

for Fmax, c, lab in zip(Fmax_list, colors, labels):
    h, F, h_eq, Fth = run_simulation(Fmax) 
    plt.plot(h * 1e9, F * 1e6, color=c, lw=2, label=lab)

plt.axhline(0, color="gray", lw=0.8)
plt.xlabel(r"Overlap, $h_{ov}$ [nm]")
plt.ylabel(r"Contact force [$\mu$N]")
plt.title("Qualitative reproduction of Fig. 2(c)")
plt.xlim(0, 120)
plt.ylim(-50 if use_cohesion else 0, 200)
plt.legend()
plt.tight_layout()
plt.show()
