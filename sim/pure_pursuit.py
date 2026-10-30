"""Degrau 3: pure pursuit. Seguir qualquer caminho, não só uma reta.

Ideia: escolher um ponto no caminho a uma distância ld à frente (a "cenoura")
e virar o volante no ângulo exato que leva o carro, num arco de círculo, até ele.

    1. achar o ponto do caminho mais perto do carro
    2. andar no caminho até achar o primeiro ponto a ≥ ld do carro (a cenoura)
    3. α = ângulo da cenoura vista do carro (0 = bem na frente)
    4. curvatura do arco que passa no carro e na cenoura: κ = 2·sen α / ld
    5. volante: δ = atan(L · κ)    (é o R = L / tan δ do degrau 1, ao contrário)

Único ajuste: ld (lookahead).

Uso: .venv/bin/python sim/pure_pursuit.py [ld ...] [--atraso s]   ex.: 0.3 1.0 2.0 --atraso 0.2

--atraso: o volante só obedece o comando s segundos depois (servo lento, sensor/rede atrasados).
Tira uma trapaça: no mundo real todo comando chega atrasado.
"""
from collections import deque
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bicicleta import DT, L, passo
from reta import DELTA_MAX

V = 1.0       # m/s: mesmo chute dos degraus anteriores
RAIO = 2.0    # m: raio das curvas da pista. Escolha minha (~2× um pátio de F1TENTH apertado)
RETA = 6.0    # m: comprimento das retas. Escolha minha
Y0 = 0.5      # m: começa 0,5 m pra dentro da pista. Escolha minha


def pista_oval(passo_m=0.05):
    """Oval no sentido anti-horário: reta de baixo → curva direita → reta de cima → curva esquerda."""
    pts = []
    for x in np.arange(0, RETA, passo_m):
        pts.append((x, 0.0))
    for a in np.arange(-np.pi / 2, np.pi / 2, passo_m / RAIO):
        pts.append((RETA + RAIO * np.cos(a), RAIO + RAIO * np.sin(a)))
    for x in np.arange(RETA, 0, -passo_m):
        pts.append((x, 2 * RAIO))
    for a in np.arange(np.pi / 2, 3 * np.pi / 2, passo_m / RAIO):
        pts.append((RAIO * np.cos(a), RAIO + RAIO * np.sin(a)))
    return np.array(pts)


def cenoura(caminho, x, y, ld):
    d = np.hypot(caminho[:, 0] - x, caminho[:, 1] - y)
    i = int(np.argmin(d))
    n = len(caminho)
    for k in range(n):  # anda pra frente no caminho (dá a volta no fim do oval)
        j = (i + k) % n
        if d[j] >= ld:
            return caminho[j], d[i]
    return caminho[i], d[i]


def pure_pursuit(x, y, th, alvo, ld):
    dx, dy = alvo[0] - x, alvo[1] - y
    # cenoura no referencial do carro (girar o mundo por −θ)
    lx = np.cos(th) * dx + np.sin(th) * dy
    ly = -np.sin(th) * dx + np.cos(th) * dy
    alpha = np.arctan2(ly, lx)
    kappa = 2 * np.sin(alpha) / ld
    return np.clip(np.arctan(L * kappa), -DELTA_MAX, DELTA_MAX)


def simular(caminho, ld, T, atraso=0.0):
    x, y, th = 0.0, Y0, 0.0
    traj, erros = [(x, y)], []
    fila = deque([0.0] * int(round(atraso / DT)))  # comandos "em trânsito"
    for _ in range(int(T / DT)):
        alvo, erro = cenoura(caminho, x, y, ld)
        fila.append(pure_pursuit(x, y, th, alvo, ld))
        delta = fila.popleft()  # o volante executa o comando mais velho
        x, y, th = passo(x, y, th, V, delta)
        traj.append((x, y))
        erros.append(erro)
    return np.array(traj), np.array(erros)


if __name__ == "__main__":
    args = sys.argv[1:]
    atraso = 0.0
    if "--atraso" in args:
        i = args.index("--atraso")
        atraso = float(args[i + 1])
        del args[i:i + 2]
    lds = [float(a) for a in args] or [1.0]  # 1 m: chute
    caminho = pista_oval()
    perimetro = 2 * RETA + 2 * np.pi * RAIO
    T = perimetro / V  # uma volta

    plt.figure(figsize=(9, 6))
    plt.plot(caminho[:, 0], caminho[:, 1], "k--", lw=1, label="caminho")
    for ld in lds:
        traj, erros = simular(caminho, ld, T, atraso)
        depois = erros[int(5 / DT):]  # ignora os 5 s iniciais (o carro começa fora)
        print(f"ld = {ld:.1f} m: erro lateral médio {depois.mean():.3f} m, máximo {depois.max():.3f} m")
        plt.plot(traj[:, 0], traj[:, 1], label=f"ld = {ld} m")
    plt.plot(0, Y0, "ko", label="início")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"Pure pursuit numa pista oval, 1 volta a 1 m/s, atraso {atraso} s")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    png = out / ("pp_ld" + "_".join(f"{ld}" for ld in lds) + f"_atraso{atraso}.png")
    plt.savefig(png, dpi=120)
    print(f"gráfico: {png}")
