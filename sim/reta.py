"""Degrau 2: seguir uma reta. Primeiro controlador.

A reta é o eixo x (y = 0). O carro começa 1 m ao lado dela, apontando reto (θ = 0).
A cada passo o controlador olha o estado e escolhe o volante δ. Velocidade fixa.

Dois controladores:
  p   δ = −k_y · y                (só o erro lateral: "tô à esquerda, vira pra direita")
  pd  δ = −k_y · y − k_θ · θ      (+ o erro de direção: "e já tô apontando pra reta?")

Uso: .venv/bin/python sim/reta.py p|pd [k_y] [k_θ]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bicicleta import DT, passo

V = 1.0          # m/s: mesmo chute do degrau 1
T = 20.0         # s: escolha minha, tempo pra ver o comportamento se repetir
DELTA_MAX = 0.4189  # rad (24°): limite do volante do F1TENTH, s_max no f1tenth_gym. Fonte.
Y0 = 1.0         # m: começa 1 m ao lado da reta. Escolha minha.


def controlador(y, th, modo, k_y, k_th):
    delta = -k_y * y
    if modo == "pd":
        delta -= k_th * th
    return np.clip(delta, -DELTA_MAX, DELTA_MAX)  # o volante tem batente


def simular(modo, k_y, k_th):
    x, y, th = 0.0, Y0, 0.0
    hist = [(0.0, x, y)]
    for i in range(int(T / DT)):
        delta = controlador(y, th, modo, k_y, k_th)
        x, y, th = passo(x, y, th, V, delta)
        hist.append(((i + 1) * DT, x, y))
    return np.array(hist)


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "p"
    k_y = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # rad por metro de erro: chute
    k_th = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0  # rad por rad de erro: chute
    assert modo in ("p", "pd"), "modo é p ou pd"

    h = simular(modo, k_y, k_th)
    t, x, y = h[:, 0], h[:, 1], h[:, 2]
    fim = y[t > T - 5]  # últimos 5 s
    nome = f"δ = −{k_y}·y" + (f" − {k_th}·θ" if modo == "pd" else "")
    print(nome)
    print(f"erro lateral nos últimos 5 s: de {fim.min():+.3f} a {fim.max():+.3f} m")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6))
    a1.plot(x, y)
    a1.axhline(0, color="k", ls="--", lw=1, label="reta")
    a1.set_xlabel("x (m)")
    a1.set_ylabel("y (m)")
    a1.set_title(f"Seguir a reta: {nome}")
    a1.legend()
    a1.grid(True)
    a2.plot(t, y)
    a2.axhline(0, color="k", ls="--", lw=1)
    a2.set_xlabel("tempo (s)")
    a2.set_ylabel("erro lateral y (m)")
    a2.grid(True)
    fig.tight_layout()
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    png = out / (f"reta_{modo}_ky{k_y}" + (f"_kth{k_th}" if modo == "pd" else "") + ".png")
    fig.savefig(png, dpi=120)
    print(f"gráfico: {png}")
