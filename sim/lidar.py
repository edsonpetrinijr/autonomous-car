"""Degrau 6: perceber a pista com LiDAR. Tira a trapaça do "caminho dado".

Até aqui o carro recebia a linha central da pista pronta. Agora ele não tem mapa:
só um LiDAR 2D que mede a distância até as paredes, em vários ângulos, várias vezes por segundo.

Controlador "centralizar entre as paredes" (o PD do degrau 2, com o erro medido pelo LiDAR):
  d_esq = menor distância nos raios da esquerda (60° a 120°)
  d_dir = menor distância nos raios da direita (−120° a −60°)
  desvio = (d_dir − d_esq) / 2           (> 0: estou mais perto da parede da esquerda)
  δ = −k_y · desvio − k_d · (variação do desvio por segundo)

O carro não sabe θ em relação à pista, então o termo D usa a variação do desvio no tempo.

Trapaça que fica: o LiDAR é perfeito (sem ruído).

Uso: .venv/bin/python sim/lidar.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bicicleta import DT, L, passo
from pure_pursuit import RETA, RAIO, Y0, cenoura, pista_oval
from reta import DELTA_MAX

V = 2.0             # m/s: mesmo do degrau 5
MEIA_PISTA = 0.75   # m: pista de 1,5 m, paredes a 0,75 m da linha central. Escolha minha
CARRO = 0.15        # m: meia largura do F1TENTH (~0,3 m). Mais perto que isso da parede = bateu
FOV = np.radians(270)    # Hokuyo UST-10LX, o LiDAR do F1TENTH: 270°. Fonte (datasheet)
N_RAIOS = 271            # 1 raio por grau. O Hokuyo tem 0,25°; reduzi pra rodar rápido no Mac
ALCANCE = 10.0           # m: alcance do UST-10LX. Fonte (datasheet)
PERIODO = 2              # LiDAR a cada 2 passos = 50 Hz. O UST-10LX faz 40 Hz. Perto o bastante
K_Y = 1.0                # mesmo k_y do degrau 2
K_D = 2 * np.sqrt(K_Y * L) / V  # ganho crítico do degrau 2 (k_θ = 2√(k_y·L)), dividido por v porque
                                # aqui o D olha a variação do desvio (= v·θ), não θ


def paredes(c=None):
    """Duas paredes paralelas à linha central c (padrão: o oval), a ±MEIA_PISTA. Viram listas de segmentos."""
    c = pista_oval() if c is None else c
    t = np.roll(c, -1, axis=0) - c
    t /= np.hypot(t[:, 0], t[:, 1])[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)  # normal à esquerda
    segs = []
    for lado in (+1, -1):
        p = c + lado * MEIA_PISTA * n
        segs.append(np.stack([p, np.roll(p, -1, axis=0)], axis=1))
    return np.concatenate(segs)  # (n_segmentos, 2 pontas, xy)


def cruz(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def escanear(x, y, th, segs):
    """Lança os raios e devolve (ângulos no referencial do carro, distâncias)."""
    angs = np.linspace(-FOV / 2, FOV / 2, N_RAIOS)
    d = np.stack([np.cos(th + angs), np.sin(th + angs)], axis=1)[:, None, :]  # (raios, 1, 2)
    a, e = segs[None, :, 0], (segs[:, 1] - segs[:, 0])[None]                  # (1, segs, 2)
    ap = a - np.array([x, y])
    den = cruz(d, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cruz(ap, e) / den   # distância ao longo do raio
        u = cruz(ap, d) / den   # onde no segmento (0 a 1)
    ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    dist = np.where(ok, t, np.inf).min(axis=1)
    return angs, np.minimum(dist, ALCANCE)


def desvio_lateral(angs, dist):
    esq = dist[(angs > np.radians(60)) & (angs < np.radians(120))].min()
    dir_ = dist[(angs < np.radians(-60)) & (angs > np.radians(-120))].min()
    return (dir_ - esq) / 2


def volta():
    segs = paredes()
    caminho = pista_oval()
    perimetro = 2 * RETA + 2 * np.pi * RAIO
    x, y, th = 0.0, Y0, 0.0
    delta, desvio_ant = 0.0, None
    traj, erros, retas, foto = [], [], [], None
    for i in range(int(perimetro / V / DT)):
        if i % PERIODO == 0:
            angs, dist = escanear(x, y, th, segs)
            if dist.min() < CARRO:
                print(f"BATEU na parede em t = {i * DT:.2f} s")
                break
            desvio = desvio_lateral(angs, dist)
            taxa = 0.0 if desvio_ant is None else (desvio - desvio_ant) / (PERIODO * DT)
            desvio_ant = desvio
            delta = np.clip(-K_Y * desvio - K_D * taxa, -DELTA_MAX, DELTA_MAX)
            if foto is None and x > RETA + 0.5:  # guarda um scan na primeira curva
                foto = (x, y, th, angs, dist)
        x, y, th = passo(x, y, th, V, delta)
        _, erro = cenoura(caminho, x, y, 1.0)
        traj.append((x, y))
        erros.append(erro)
        retas.append(0 <= x <= RETA)  # nas retas, x fica entre 0 e RETA
    return np.array(traj), np.array(erros), np.array(retas), foto, segs


if __name__ == "__main__":
    traj, erros, retas, foto, segs = volta()
    ini = int(len(erros) * 0.2)
    e, r = erros[ini:], retas[ini:]
    print(f"erro lateral nas retas : médio {e[r].mean() * 100:5.1f} cm, máx {e[r].max() * 100:5.1f} cm")
    print(f"erro lateral nas curvas: médio {e[~r].mean() * 100:5.1f} cm, máx {e[~r].max() * 100:5.1f} cm")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for s in segs[::4]:
        a1.plot(s[:, 0], s[:, 1], "k-", lw=0.8)
    c = pista_oval()
    a1.plot(c[:, 0], c[:, 1], "k:", lw=0.8, label="linha central (o carro NÃO sabe)")
    a1.plot(traj[:, 0], traj[:, 1], lw=1.5, label="carro")
    a1.axis("equal")
    a1.grid(True)
    a1.legend(fontsize=8)
    a1.set_title("Centralizar entre as paredes, só com LiDAR")

    if foto:
        x, y, th, angs, dist = foto
        px, py = dist * np.cos(angs), dist * np.sin(angs)  # pontos no referencial do carro
        a2.plot(px, py, ".", ms=2, label="o que o LiDAR vê")
        a2.plot(0, 0, "r^", ms=10, label="carro (olhando pra +x)")
        a2.axis("equal")
        a2.grid(True)
        a2.legend(fontsize=8)
        a2.set_title("Um scan, entrando na 1ª curva (referencial do carro)")
    fig.tight_layout()
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    png = out / "lidar.png"
    fig.savefig(png, dpi=120)
    print(f"gráfico: {png}")
