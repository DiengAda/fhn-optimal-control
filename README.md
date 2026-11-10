# Contrôle optimal du modèle de FitzHugh–Nagumo et réseaux de neurones

Projet de Master 2 MIGS (Université Bourgogne Europe), encadré par Cristian Mendico, 2026–2027.

## Objectifs

1. Faire la correspondance entre le modèle de FitzHugh–Nagumo (FHN) et la théorie du contrôle optimal : état, commande, coût, conditions d'optimalité (principe de Pontryagin).
2. Comprendre et reproduire les méthodes numériques de Lou & Swamy (2017).
3. Reproduire les expériences numériques avec des réseaux de neurones (PyTorch).
4. Bonus : fonction valeur, programmation dynamique, équation de Hamilton–Jacobi.

## Le modèle

FHN adimensionné (Lou & Swamy, éq. 30) :

```
delta * dv/dt = v (v + 0.6)(1 - v) - w + I
        dw/dt = v - 0.5 w
```

v : potentiel de membrane (rapide), w : variable de récupération (lente), I : courant injecté (commande).

## Structure du dépôt

```
simulations/   scripts de simulation (FHN sans contrôle, contrôle optimal, ...)
src/           code réutilisable (modèles, solveurs, réseaux)
figures/       figures générées par les scripts
notes/         notes théoriques et comptes rendus
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python simulations/fhn_fig1.py   # reproduit la Fig. 1 de Lou & Swamy (2017)
```

## Références

- X. Lou, M.N.S. Swamy, *A new approach to optimal control of conductance-based spiking neurons*, Neural Networks 96 (2017) 128–136.
- W. Kang, Q. Gong, T. Nakamura-Zimmerer, F. Fahroo, *Algorithms of data generation for deep learning and feedback design: a survey*, arXiv:1912.00492.
