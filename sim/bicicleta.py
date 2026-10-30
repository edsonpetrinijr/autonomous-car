"""Degrau 1: o carro sozinho. Modelo de bicicleta cinemática.

Trapaças (de propósito): a posição é perfeita, não tem sensor, o pneu não escorrega.

Estado:   x, y (m) e θ (rad), medidos no eixo TRASEIRO.
Entradas: v (m/s) e δ (rad, ângulo do volante na roda da frente).

    dx/dt = v · cos θ
    dy/dt = v · sin θ
    dθ/dt = v / L · tan δ

Uso: .venv/bin/python sim/bicicleta.py [delta_rad] [v_m_s]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

L = 0.33   # entre-eixos (m): carro F1TENTH, wheelbase 0.3302 m nos parâmetros do f1tenth_gym. Fonte, não chute.
DT = 0.01  # passo de integração (s): escolha minha, pequeno o bastante pro erro do Euler ser < 1%
T = 10.0   # duração (s): escolha minha, dá mais de uma volta nos valores padrão


def passo(x, y, th, v, delta):
    """Avança o carro DT segundos (Euler: derivada × DT)."""
    return (x + v * np.cos(th) * DT,
            y + v * np.sin(th) * DT,
            th + v / L * np.tan(delta) * DT)


def simular(delta, v):
    x, y, th = 0.0, 0.0, 0.0  # começa na origem, apontando pro +x
    traj = [(x, y)]
    for _ in range(int(T / DT)):
        x, y, th = passo(x, y, th, v, delta)
        traj.append((x, y))
    return np.array(traj)


def raio_medido(traj):
    # a altura do círculo (y máx − y mín) é o diâmetro
    return (traj[:, 1].max() - traj[:, 1].min()) / 2


if __name__ == "__main__":
    delta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3  # 0.3 rad ≈ 17°: chute, volante bem virado
    v = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0      # 1 m/s: chute, passo de gente andando

    traj = simular(delta, v)
    r_med = raio_medido(traj)
    r_teo = L / np.tan(delta)
    print(f"δ = {delta:.2f} rad ({np.degrees(delta):.0f}°), v = {v} m/s")
    print(f"raio medido  = {r_med:.3f} m")
    print(f"raio teórico = {r_teo:.3f} m   (L / tan δ)")
    print(f"diferença    = {abs(r_med - r_teo) / r_teo * 100:.2f} %")
    assert abs(r_med - r_teo) / r_teo < 0.02, "simulação não bate com a teoria"

    plt.plot(traj[:, 0], traj[:, 1])
    plt.plot(0, 0, "ko", label="início")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"Bicicleta cinemática: δ = {delta:.2f} rad, v = {v} m/s, {T:.0f} s")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    png = out / f"bicicleta_d{delta:.2f}_v{v}.png"
    plt.savefig(png, dpi=120)
    print(f"gráfico: {png}")
