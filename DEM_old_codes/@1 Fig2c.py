import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters

E_eff   = 13.2e9
alpha   = 0.65
R_eff   = 8e-7
sigma_H = 1.9e9
t_rel   = 2.0e-6
phis    = 0.4

k_coh   = 1.0
F_tensile_max = 1

n_steps = 20000
t_total = 1.00
dt      = t_total/n_steps

F_maxs  = [200e-6, 100e-6, 50e-6]

# -----------------------------

def k_spring(h):
    return (4.0/3.0) * E_eff * np.sqrt(max(h, 0.0) * R_eff)

def F_n_comp(h, heq):
    if h <= 0.0: return 0.0
    return k_spring(h) * max(h - heq, 0.0)

def F_tensile(h, heq):
    if h >= heq: return 0.0
    k_ref = k_spring(max(heq, 1e-12))
    Ft = -k_coh * k_ref * (heq - h)
    return max(Ft, -F_tensile_max)

def F_th(h):
    return (2.0/3.0) * sigma_H * np.pi * R_eff * alpha * h

def dh_eq_per_dt(h, heq):
    temp = max(F_n_comp(h, heq) - F_th(h), 0)/(t_rel * k_spring(h))*(1-phis**4)
    return temp

def F_tot(h, heq):
    if h >= heq: return F_n_comp(h, heq)
    return F_tensile(h, heq)

def solve_h(F_app, heq):
    """
    For F_app > 0 : Compressive case
    For F_app < 0 : Tensile case
    """
    
    # compressive case
    if F_app >= 0:
        lo  = max(heq, 0.0)
        hi  = max(lo+1e-12, 1e-9)
        while F_tot(hi, heq) < F_app:
            hi *= 1.5
            if hi > 1e-3:
                raise RuntimeError("Faield to bracket compressive root")
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if F_tot(mid, heq) < F_app: lo = mid
            else:   hi = mid
        return 0.5 * (lo + hi)
    
    # tensile case
    hi = max(heq, 0.0)
    lo = max(0.0, hi - 1e-9)
    
    while F_tot(lo, heq) > F_app:
        span = hi - lo
        lo  = max(0.0, lo - max(1e-12, 1.5 * span))
        if lo <= 0.0 and F_tot(lo, heq) > F_app: return 0.0
    
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if F_tot(mid, heq) > F_app: hi = mid
        else:   lo = mid
    return 0.5 * (lo + hi)

def Force_injection(Fmax, n_steps):
    ckpt = n_steps*2//5
    
    up = np.linspace(0.0, Fmax, ckpt, endpoint = False)
    down = np.linspace(Fmax, -1.5 * Fmax, n_steps - ckpt)
    
    return np.concatenate([up, down])

# def Force_injection(Fmax, n_steps):
#     ckpt = n_steps//2
    
#     up = np.linspace(0.0, Fmax, ckpt, endpoint = False)
#     down = np.linspace(Fmax, 0.0, n_steps - ckpt)
    
#     return np.concatenate([up, down])

def run_simulation(Fmax):
    F_hist = Force_injection(Fmax, n_steps)
    
    heq = 0.0
    h_list = []
    F_list = []
    heq_list = []
    Fth_list = []
    
    for F_app in F_hist:
        h = solve_h(F_app, heq)
        
        if h > 0 :
            F_now   = F_n_comp(h, heq)
            Fth     = F_th(h)
            ks      = max(k_spring(h), 1e-30)
            
            if F_now > Fth:
                dheq = ((F_now - Fth) / (t_rel * ks)) * dt
                heq = min(heq + max(dheq, 0.0), h)
            Fth_list.append(Fth)
        else:
            Fth_list.append(0.0)
        
        h_list.append(h)
        heq_list.append(heq)
        F_list.append(F_app)
    
    return np.array(h_list), np.array(F_list), np.array(heq_list), np.array(Fth_list)

# Run

plt.figure(figsize=(7, 5))

colors = ["goldenrod", "tab:orange", "tab:blue"]
labels = [
    r"$F_{\max}=200\ \mu N$",
    r"$F_{\max}=100\ \mu N$",
    r"$F_{\max}=50\ \mu N$",
]

# Hertzian reference (h_eq = 0)
h_ref = np.linspace(0, 120e-9, 400)
F_ref = np.array([F_n_comp(h, 0.0) for h in h_ref])
plt.plot(h_ref * 1e9, F_ref * 1e6, "k--", lw=1.5, label="Hertzian contact law")

for Fmax, c, lab in zip(F_maxs, colors, labels):
    h, F, h_eq, Fth = run_simulation(Fmax) 
    plt.plot(h * 1e9, F * 1e6, color=c, lw=2, label=lab)

plt.axhline(0, color="gray", lw=0.8)
plt.xlabel(r"Overlap, $h_{ov}$ [nm]")
plt.ylabel(r"Contact force [$\mu$N]")
plt.title("Reproduction of Fig. 2(c)")
plt.xlim(0, 120)
plt.ylim(-50, 200)
plt.legend()
plt.tight_layout()
plt.show()



