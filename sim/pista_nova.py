"""Meu modelo, passo 1b: ele aprendeu a dirigir ou só decorou a pista?

A rede treinada no oval (sim/imitacao.py) dirige numa pista que ela nunca viu:
um "feijão" com curvas de raios diferentes e, ao contrário do oval, trechos virando pra DIREITA.
No oval, todas as curvas eram pra esquerda; a rede nunca viu o especialista virar pra direita.

Precisa rodar sim/imitacao.py antes (ele salva sim/out/motorista.pt).
Uso: .venv/bin/python sim/pista_nova.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from imitacao import Motorista, V, dirigir, especialista, rede_politica
from lidar import paredes


def pista_feijao(passo_m=0.05):
    """Curva fechada r(φ) = 4 + 0,7·sen(2φ) + 0,6·cos(3φ), reamostrada a cada 5 cm. Números ajustados pra ter
    curva pros dois lados (raio mínimo ~2,1 m à esquerda e ~1,5 m à direita) sem as paredes se cruzarem."""
    phi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    r = 4 + 0.7 * np.sin(2 * phi) + 0.6 * np.cos(3 * phi)
    p = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0, append=p[:1]).T))])
    novo = np.arange(0, s[-1], passo_m)
    fechado = np.vstack([p, p[:1]])
    return np.stack([np.interp(novo, s, fechado[:, 0]), np.interp(novo, s, fechado[:, 1])], axis=1)


def curvatura(c):
    d = np.roll(c, -1, axis=0) - c
    rumo = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.gradient(rumo) / np.hypot(d[:, 0], d[:, 1])


if __name__ == "__main__":
    caminho = pista_feijao()
    segs = paredes(caminho)
    k = curvatura(caminho)[2:-2]  # sem as pontas: a emenda do fim com o começo dá um pico falso
    perimetro = len(caminho) * 0.05
    print(f"pista nova: {perimetro:.1f} m | curva mais fechada: esquerda R = {1 / k.max():.1f} m, "
          f"direita R = {1 / -k.min():.1f} m | {(k < 0).mean() * 100:.0f}% do caminho virando pra direita")

    out = Path(__file__).parent / "out"
    pilotos = [("especialista", especialista(caminho))]
    for arq, nome in (("motorista.pt", "rede (oval)"), ("motorista_espelhado.pt", "rede (oval + espelho)")):
        if (out / arq).exists():
            modelo = Motorista(28)
            modelo.load_state_dict(torch.load(out / arq))
            modelo.eval()
            pilotos.append((nome, rede_politica(modelo)))

    d0 = caminho[1] - caminho[0]
    x0, y0, th0 = caminho[0][0], caminho[0][1], float(np.arctan2(d0[1], d0[0]))
    k_todo = curvatura(caminho)
    plt.figure(figsize=(8, 7))
    for s in segs[::4]:
        plt.plot(s[:, 0], s[:, 1], "k-", lw=0.6)
    for quem, pol in pilotos:
        traj, _, _, erros, bateu = dirigir(pol, x0, y0, th0, perimetro / V, segs, caminho)
        # erro máximo só nos trechos virando pra direita (o que a rede do oval nunca viu)
        perto = [int(np.argmin(np.hypot(caminho[:, 0] - px, caminho[:, 1] - py))) for px, py in traj]
        direita = k_todo[perto] < 0
        dir_max = erros[direita].max() * 100 if direita.any() else float("nan")
        status = (f"BATEU depois de {len(traj) * 0.01:.1f} s" if bateu
                  else f"erro médio {erros.mean() * 100:.1f} cm, máx {erros.max() * 100:.1f} cm, "
                       f"máx nas curvas pra direita {dir_max:.1f} cm")
        print(f"{quem:>24}: {status}")
        plt.plot(traj[:, 0], traj[:, 1], lw=1.5, label=quem + (" (BATEU)" if bateu else ""))
        if bateu:
            plt.plot(*traj[-1], "rx", ms=12, mew=3)
    plt.plot(x0, y0, "ko", label="largada")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.title("Pista que a rede nunca viu")
    plt.savefig(out / "pista_nova.png", dpi=120)
    print(f"gráfico: {out / 'pista_nova.png'}")
