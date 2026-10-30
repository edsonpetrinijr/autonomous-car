"""Vídeo estilo Deep Cars: os 200 carros de cada geração dirigindo juntos no oval.

Precisa rodar sim/evolucao.py antes (salva sim/out/evolucao_populacoes.npz).
Uso: .venv/bin/python sim/video_evolucao.py [--limpo]   → sim/out/evolucao[_limpo].mp4
"""
import sys
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evolucao import A_CADA, DT, T, segmentos, simular
from pure_pursuit import Y0, pista_oval

SUF = "_limpo" if "--limpo" in sys.argv else ""
FPS = int(round(1 / (DT * A_CADA)))  # 1 quadro por decisão da rede = 20 fps = tempo real


def main():
    out = Path(__file__).parent / "out"
    pops = np.load(out / f"evolucao_populacoes{SUF}.npz")
    oval = pista_oval()
    segs = segmentos(oval)
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=110)
    escritor = imageio.get_writer(out / f"evolucao{SUF}.mp4", fps=FPS, macro_block_size=1)
    for chave in sorted(pops.files, key=lambda k: int(k[1:])):
        g = int(chave[1:])
        _, _, _, tr, vivos = simular(pops[chave], oval, segs, T, (0.0, Y0, 0.0), guardar=True)
        print(f"geração {g}: {len(tr)} quadros")
        for q in range(len(tr)):
            ax.clear()
            for s in segs:
                ax.plot(s[:, 0], s[:, 1], "k-", lw=1)
            v = vivos[q]
            ax.scatter(tr[q, ~v, 0], tr[q, ~v, 1], s=8, c="lightgray")          # batidos
            ax.scatter(tr[q, v, 0], tr[q, v, 1], s=10, c="tab:blue", alpha=0.6)  # vivos
            ax.plot(tr[: q + 1, 0, 0], tr[: q + 1, 0, 1], "r-", lw=1)           # rastro do campeão
            ax.scatter(*tr[q, 0], s=40, c="red", zorder=5)
            ax.set_xlim(-3, 9)
            ax.set_ylim(-1.5, 5.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"Geração {g}   |   vivos: {v.sum()} de {len(v)}   |   t = {q / FPS:4.1f} s"
                         "\n(vermelho = o melhor carro da geração)", fontsize=12)
            fig.canvas.draw()
            escritor.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    escritor.close()
    print(f"vídeo: {out / f'evolucao{SUF}.mp4'}")


if __name__ == "__main__":
    main()
