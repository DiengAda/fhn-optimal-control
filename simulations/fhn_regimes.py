"""
Regimes dynamiques du modele de FitzHugh-Nagumo (forme de Lou & Swamy, eq. 30) :

    delta * dv/dt = v (v + a)(1 - v) - w + I        (a = 0.6)
            dw/dt = v - 0.5 w

Pour chaque jeu (I, delta) on calcule l'equilibre, sa stabilite (valeurs propres
de la jacobienne), puis on simule. Quatre cas :
  A. I = 0.2,  delta = 0.01 : oscillations (Fig. 1 du papier)
  B. I = 0,    delta = 0.01 : oscillations autour de l'origine (origine instable)
  C. I = 0,    delta = 1    : cas de l'experience de controle du papier
  D. I = -0.8, delta = 0.01 : regime excitable (equilibre stable + seuil)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

A = 0.6


# ---------------------------------------------------------------
# 1. Modele, equilibre, stabilite
# ---------------------------------------------------------------
def cubic(v, a=A):
    """Terme non lineaire v (v + a)(1 - v)."""
    return v * (v + a) * (1 - v)


def dcubic(v, a=A):
    """Derivee du terme cubique : -3 v^2 + 2(1 - a) v + a."""
    return -3 * v**2 + 2 * (1 - a) * v + a


def fhn(t, x, I, delta):
    v, w = x
    return [(cubic(v) - w + I) / delta, v - 0.5 * w]


def equilibre(I):
    """Intersection des isoclines : w = 2v et cubic(v) - w + I = 0
    => v^3 - (1 - a) v^2 + (2 - a) v - I = 0 (une seule racine reelle)."""
    r = np.roots([1, -(1 - A), 2 - A, -I])
    v = r[np.isreal(r)].real[0]
    return np.array([v, 2 * v])


def jacobienne(x, delta):
    v, _ = x
    return np.array([[dcubic(v) / delta, -1 / delta],
                     [1.0, -0.5]])


def simuler(I, delta, x0, T, n=20001):
    t = np.linspace(0, T, n)
    sol = solve_ivp(fhn, (0, T), x0, args=(I, delta), method="Radau",
                    t_eval=t, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


# ---------------------------------------------------------------
# 2. Les quatre cas
# ---------------------------------------------------------------
cas = {
    "A": dict(I=0.2,  delta=0.01, T=10,  titre="A. I = 0.2, δ = 0.01 (Fig. 1)"),
    "B": dict(I=0.0,  delta=0.01, T=10,  titre="B. I = 0, δ = 0.01"),
    "C": dict(I=0.0,  delta=1.0,  T=40,  titre="C. I = 0, δ = 1 (expérience de contrôle)"),
    "D": dict(I=-0.8, delta=0.01, T=10,  titre="D. I = -0.8, δ = 0.01 (excitable)"),
}

print(f"{'cas':<4}{'equilibre (v, w)':<22}{'valeurs propres':<36}{'nature'}")
for nom, c in cas.items():
    xe = equilibre(c["I"])
    lam = np.linalg.eigvals(jacobienne(xe, c["delta"]))
    nature = "STABLE" if np.all(lam.real < 0) else "INSTABLE"
    c["xe"], c["lam"] = xe, lam
    lam_txt = ", ".join(f"{l.real:.2f}{l.imag:+.2f}i" for l in lam)
    print(f"{nom:<4}({xe[0]:+.3f}, {xe[1]:+.3f}){'':<6}{lam_txt:<36}{nature}")

# Conditions initiales
cas["A"]["x0s"] = [[0.32, 0.01]]                        # celles du papier
cas["B"]["x0s"] = [[0.01, 0.0]]                         # tout pres de l'origine
cas["C"]["x0s"] = [[1.0, -0.1], [0.01, 0.0]]            # x0 du papier + pres de l'origine

# Cas D : seuil = branche du milieu de la cubique a w = w_eq
xe = cas["D"]["xe"]
v_seuil = brentq(lambda v: cubic(v) - xe[1] + cas["D"]["I"], xe[0] + 1e-3, 0.6)
print(f"\nCas D : repos v = {xe[0]:.3f}, seuil v ~ {v_seuil:.3f} "
      f"(perturbation critique ~ {v_seuil - xe[0]:.3f})")
dv_c = v_seuil - xe[0]
cas["D"]["x0s"] = [[xe[0] + 0.8 * dv_c, xe[1]],        # sous le seuil
                   [xe[0] + 1.2 * dv_c, xe[1]]]        # au-dessus du seuil

for c in cas.values():
    c["sols"] = [simuler(c["I"], c["delta"], x0, c["T"]) for x0 in c["x0s"]]


# ---------------------------------------------------------------
# 3. Figure 1 : v(t) pour chaque cas
# ---------------------------------------------------------------
styles = ["b-", "m-"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (nom, c) in zip(axes.flat, cas.items()):
    for k, (t, (v, w)) in enumerate(c["sols"]):
        lab = f"v, x0 = ({c['x0s'][k][0]:.2f}, {c['x0s'][k][1]:.2f})"
        ax.plot(t, v, styles[k], lw=1.2, label=lab)
        ax.plot(t, w, styles[k][0] + "--", lw=0.9, alpha=0.6)
    ax.set_title(c["titre"], fontsize=10)
    ax.set_xlabel("t"); ax.set_ylabel("v (plein), w (pointillé)")
    ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(FIG_DIR / "regimes_series_temporelles.png", dpi=150)


# ---------------------------------------------------------------
# 4. Figure 2 : plans de phase
# ---------------------------------------------------------------
vv = np.linspace(-1, 1.3, 400)
fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9))
for ax, (nom, c) in zip(axes2.flat, cas.items()):
    ax.plot(vv, cubic(vv) + c["I"], "k-", lw=0.9, label=r"$\dot v = 0$")
    ax.plot(vv, 2 * vv, "k--", lw=0.9, label=r"$\dot w = 0$")
    for k, (t, (v, w)) in enumerate(c["sols"]):
        ax.plot(v, w, styles[k], lw=1.1)
        ax.plot(*c["x0s"][k], styles[k][0] + "s", ms=4)
    couleur = "go" if np.all(c["lam"].real < 0) else "ro"
    ax.plot(*c["xe"], couleur, ms=7,
            label="équilibre " + ("stable" if couleur == "go" else "instable"))
    ax.set_xlim(-1, 1.3); ax.set_ylim(-1.1, 1)
    ax.set_title(c["titre"], fontsize=10)
    ax.set_xlabel("v"); ax.set_ylabel("w")
    ax.legend(fontsize=7, loc="upper left")
fig2.tight_layout()
fig2.savefig(FIG_DIR / "regimes_plans_de_phase.png", dpi=150)
print("\nFigures enregistrees dans", FIG_DIR)
