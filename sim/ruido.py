"""Degrau 5: tira a trapaça da "posição perfeita".

Até aqui o controlador sabia exatamente onde o carro estava. No mundo real, a posição
vem de um sensor (GPS, LiDAR + mapa, câmera) e sempre tem ruído. O carro de verdade
continua andando certinho; quem erra é o que o controlador ACHA da posição.

Quatro jeitos de usar a medida:
  verdade   o controlador sabe a posição real (trapaça, só pra comparar)
  cru       usa a medida com ruído direto
  media     média das últimas N medidas
  filtro    prever e corrigir: anda a estimativa com o modelo de bicicleta (sei v e δ),
            e puxa um pouco (K) em direção a cada medida nova. Ideia central do filtro de Kalman.

A direção θ continua perfeita (como se uma IMU boa desse ela). Trapaça que fica.

Uso: .venv/bin/python sim/ruido.py
"""
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bicicleta import DT, passo
from pure_pursuit import RETA, RAIO, Y0, cenoura, pista_oval, pure_pursuit

V = 2.0        # m/s: bem abaixo do limite de aderência do degrau 4 (4,5), pra isolar o efeito do ruído
LD = 1.0       # m: o mesmo do degrau 4
SIGMA = 0.15   # m: desvio-padrão do ruído na posição. Chute (GPS comum ~2 m; LiDAR + mapa ~5 cm)
N_MEDIA = 30   # medidas na média (0,3 s a 100 Hz). Chute
K = 0.05       # quanto o filtro confia em cada medida nova (0 = ignora, 1 = só medida). Chute


def volta(modo, seed=0):
    rng = np.random.default_rng(seed)  # mesma sequência de ruído em todos os modos: comparação justa
    caminho = pista_oval()
    perimetro = 2 * RETA + 2 * np.pi * RAIO
    x, y, th = 0.0, Y0, 0.0          # onde o carro ESTÁ
    ex, ey = x, y                    # onde o controlador ACHA que está
    janela = deque(maxlen=N_MEDIA)
    delta = 0.0
    real, achado, erros, deltas = [], [], [], []
    for _ in range(int(perimetro / V / DT)):
        mx, my = x + rng.normal(0, SIGMA), y + rng.normal(0, SIGMA)  # o sensor
        if modo == "verdade":
            ex, ey = x, y
        elif modo == "cru":
            ex, ey = mx, my
        elif modo == "media":
            janela.append((mx, my))
            ex, ey = np.mean(janela, axis=0)
        elif modo == "filtro":
            ex, ey, _ = passo(ex, ey, th, V, delta)  # prever: onde devo estar agora
            ex += K * (mx - ex)                       # corrigir: puxa um pouco pra medida
            ey += K * (my - ey)
        alvo, _ = cenoura(caminho, ex, ey, LD)
        delta = pure_pursuit(ex, ey, th, alvo, LD)
        x, y, th = passo(x, y, th, V, delta)
        _, erro_real = cenoura(caminho, x, y, LD)
        real.append((x, y))
        achado.append((ex, ey))
        erros.append(erro_real)
        deltas.append(delta)
    return np.array(real), np.array(achado), np.array(erros), np.array(deltas)


if __name__ == "__main__":
    modos = ["verdade", "cru", "media", "filtro"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    caminho = pista_oval()
    for ax, modo in zip(axs.flat, modos):
        real, achado, erros, deltas = volta(modo)
        e = erros[int(len(erros) * 0.2):]
        tremor = np.degrees(np.abs(np.diff(deltas))).mean()
        print(f"{modo:>8}: erro lateral médio {e.mean() * 100:5.1f} cm, máx {e.max() * 100:5.1f} cm"
              f" | volante tremendo {tremor:5.2f}°/passo")
        ax.plot(caminho[:, 0], caminho[:, 1], "k--", lw=1)
        if modo != "verdade":
            ax.plot(achado[:, 0], achado[:, 1], ".", ms=1, alpha=0.4, label="onde ACHA que está")
        ax.plot(real[:, 0], real[:, 1], lw=1.5, label="onde ESTÁ")
        ax.set_title(f"{modo}: erro médio {e.mean() * 100:.1f} cm")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.suptitle(f"Posição com ruído σ = {SIGMA} m, {V} m/s")
    fig.tight_layout()
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    png = out / "ruido.png"
    fig.savefig(png, dpi=120)
    print(f"gráfico: {png}")
