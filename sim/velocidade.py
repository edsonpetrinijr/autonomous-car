"""Degrau 4: velocidade. Tira a trapaça do "pneu nunca escorrega".

Pra fazer uma curva de raio R a v, o carro precisa de aceleração lateral a = v² / R.
Quem dá essa força é o atrito do pneu, e ele tem limite: a ≤ μ·g.
Se o controlador pedir uma curva mais fechada que isso, o pneu escorrega e o carro
faz a curva mais aberta possível (κ = μ·g / v²): "sai de frente" (understeer).

Dois modos:
  const v1 v2 ...   velocidade constante a volta toda
  perfil            rápido na reta, devagar na curva: v = min(V_MAX, √(μ·g / |κ|))

Uso: .venv/bin/python sim/velocidade.py const 1 3 5
     .venv/bin/python sim/velocidade.py perfil
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bicicleta import DT, L
from pure_pursuit import RETA, RAIO, Y0, cenoura, pista_oval, pure_pursuit

MU = 1.0489     # atrito pneu-chão: parâmetro mu do f1tenth_gym. Fonte (medido no carro deles)
G = 9.81        # m/s²
A_MAX = MU * G  # ≈ 10,3 m/s²: aceleração lateral máxima antes de escorregar
LD = 1.0        # m: melhor lookahead do degrau 3 com atraso
V_MAX = 8.0     # m/s: velocidade máxima na reta. Chute (F1TENTH real anda ~8 m/s)
MEIA_PISTA = 0.5  # m: pista de 1 m de largura; erro > 0,5 m = saiu. Escolha minha


def curvatura(caminho):
    """κ em cada ponto = quanto a direção do caminho muda por metro."""
    d = np.diff(caminho, axis=0, append=caminho[:1])
    rumo = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    ds = np.hypot(d[:, 0], d[:, 1])
    return np.gradient(rumo) / ds


def volta(caminho, vel_no_ponto):
    """Uma volta. vel_no_ponto(i) = velocidade quando o ponto mais perto é o i."""
    perimetro = 2 * RETA + 2 * np.pi * RAIO
    x, y, th, t, andou = 0.0, Y0, 0.0, 0.0, 0.0
    traj, erros, escorregou = [(x, y)], [], 0
    while andou < perimetro and t < 120:
        i = int(np.argmin(np.hypot(caminho[:, 0] - x, caminho[:, 1] - y)))
        v = vel_no_ponto(i)
        alvo, erro = cenoura(caminho, x, y, LD)
        kappa = np.tan(pure_pursuit(x, y, th, alvo, LD)) / L  # curva que o controlador pede
        k_max = A_MAX / v**2                                   # curva que o pneu aguenta
        if abs(kappa) > k_max:
            kappa = np.sign(kappa) * k_max
            escorregou += 1
        x += v * np.cos(th) * DT
        y += v * np.sin(th) * DT
        th += v * kappa * DT
        t += DT
        andou += v * DT
        traj.append((x, y))
        erros.append(erro)
    return np.array(traj), np.array(erros), t, escorregou * DT


def relatorio(nome, traj, erros, t, escorregou):
    e = erros[int(len(erros) * 0.2):]  # ignora o começo (o carro sai de fora da linha)
    saiu = "SAIU DA PISTA" if e.max() > MEIA_PISTA else "ok"
    print(f"{nome:>14}: volta {t:5.2f} s | erro máx {e.max():.2f} m | escorregou {escorregou:4.2f} s | {saiu}")
    plt.plot(traj[:, 0], traj[:, 1], label=f"{nome}: {t:.2f} s")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "const"
    caminho = pista_oval()
    kappa = curvatura(caminho)

    plt.figure(figsize=(9, 6))
    plt.plot(caminho[:, 0], caminho[:, 1], "k--", lw=1, label="caminho")
    if modo == "const":
        for v in [float(a) for a in sys.argv[2:]] or [1.0, 3.0, 5.0]:
            relatorio(f"{v} m/s", *volta(caminho, lambda i, v=v: v))
    else:
        v_ponto = np.minimum(V_MAX, np.sqrt(A_MAX / np.maximum(np.abs(kappa), 1e-6)))
        relatorio("perfil", *volta(caminho, lambda i: v_ponto[i]))
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"Velocidade × aderência (μ = {MU}), ld = {LD} m")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    png = out / f"vel_{modo}_{'_'.join(sys.argv[2:]) or 'padrao'}.png"
    plt.savefig(png, dpi=120)
    print(f"gráfico: {png}")
