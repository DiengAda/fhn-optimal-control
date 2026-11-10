"""
Reproduction de la Fig. 1 de Lou & Swamy (2017) :
oscillations du modele de FitzHugh-Nagumo (FHN) sans controle.

Modele (eq. 30) :
    delta * dv/dt = v (v + 0.6) (1 - v) - w + I
            dw/dt = v - 0.5 w
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. Parametres (ceux de la Fig. 1)
# ---------------------------------------------------------------
a = 0.6        # parametre du terme cubique
delta = 0.01   # separation des echelles de temps (v rapide, w lent)
I = 0.2        # courant constant injecte
x0 = [0.32, 0.01]   # conditions initiales (v(0), w(0))
T = 10.0            # temps final


# ---------------------------------------------------------------
# 2. Le second membre : x' = F(x) + B I
# ---------------------------------------------------------------
def fhn(t, x, I=I, delta=delta, a=a):
    v, w = x
    dv = (v * (v + a) * (1 - v) - w + I) / delta   # on divise TOUT par delta
    dw = v - 0.5 * w
    return [dv, dw]


# ---------------------------------------------------------------
# 3. Integration numerique
# ---------------------------------------------------------------
t_eval = np.linspace(0, T, 20001)          # grille fine pour bien voir les fronts
sol = solve_ivp(fhn, (0, T), x0,
                method="Radau",            # solveur implicite : adapte aux systemes raides
                t_eval=t_eval, rtol=1e-8, atol=1e-10)
t, (v, w) = sol.t, sol.y


# ---------------------------------------------------------------
# 4. Verifications chiffrees
# ---------------------------------------------------------------
pics = np.where((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > 0.5))[0] + 1
print(f"Nombre de pics de v : {len(pics)}")
print(f"Periode moyenne     : {np.diff(t[pics]).mean():.3f}")
print(f"v dans [{v.min():.2f}, {v.max():.2f}],  w dans [{w.min():.2f}, {w.max():.2f}]")


# ---------------------------------------------------------------
# 5. Figure 1 : v(t) et w(t)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t, v, "b-", lw=1.3, label="v")
ax.plot(t, w, "r--", lw=1.3, label="w")
ax.set_xlim(0, T)
ax.set_ylim(-1, 1.5)
ax.set_xlabel("t")
ax.set_ylabel("v, w")
ax.legend(loc="upper right")
ax.set_title(r"FHN : $\delta=0.01$, $I=0.2$, $v(0)=0.32$, $w(0)=0.01$")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_fhn_oscillations.png", dpi=150)


# ---------------------------------------------------------------
# 6. Bonus : plan de phase et isoclines nulles
# ---------------------------------------------------------------
vv = np.linspace(-1, 1.3, 400)
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(vv, vv * (vv + a) * (1 - vv) + I, "k-", lw=1, label=r"$\dot v = 0$ (cubique)")
ax2.plot(vv, 2 * vv, "k--", lw=1, label=r"$\dot w = 0$ ($w = 2v$)")
ax2.plot(v, w, "b-", lw=1.2, label="trajectoire")
# equilibre : intersection des deux isoclines
racines = np.roots([-1, 1 - a, a - 2, I])           # -v^3 + 0.4 v^2 - 1.4 v + 0.2 = 0
v_eq = racines[np.isreal(racines)].real[0]
ax2.plot(v_eq, 2 * v_eq, "ro", label=f"equilibre ({v_eq:.3f}, {2*v_eq:.3f})")
ax2.set_xlim(-1, 1.3); ax2.set_ylim(-0.4, 0.8)
ax2.set_xlabel("v"); ax2.set_ylabel("w")
ax2.legend(fontsize=8, loc="upper left")
ax2.set_title("Plan de phase : cycle limite autour de l'equilibre instable")
fig2.tight_layout()
fig2.savefig(FIG_DIR / "fig1_bis_plan_de_phase.png", dpi=150)
print(f"Equilibre : v = {v_eq:.3f}, w = {2*v_eq:.3f}")
