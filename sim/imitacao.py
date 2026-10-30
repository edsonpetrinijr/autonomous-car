"""Meu modelo que dirige, passo 1: imitação (behavior cloning).

1. COLETAR: o especialista (pure pursuit, que conhece o mapa e a posição exata) dá algumas voltas.
   A cada leitura do LiDAR, grava o par (o que o LiDAR viu, o volante que o especialista usou).
2. TREINAR: uma rede neural pequena aprende a função  scan do LiDAR → volante.
3. TESTAR: a rede dirige sozinha. Ela só tem o LiDAR: sem mapa, sem posição, sem regra escrita.

Uso: .venv/bin/python sim/imitacao.py [--espelhar]

--espelhar: data augmentation. Cada scan também entra invertido (esquerda ↔ direita) com o volante
trocado de sinal. Toda curva pra esquerda do oval vira uma curva pra direita, sem coletar nada novo.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from bicicleta import DT, passo
from lidar import CARRO, PERIODO, escanear, paredes
from pure_pursuit import RETA, RAIO, cenoura, pista_oval, pure_pursuit
from reta import DELTA_MAX

V = 2.0           # m/s: o mesmo dos degraus 5 e 6
LD = 1.0          # m: lookahead do especialista
A_CADA = 10       # usa 1 raio a cada 10 (271 → 28): rede menor, treino rápido. Escolha minha
CORTE = 5.0       # m: distâncias acima disso viram 5. Longe não importa pra virar agora. Escolha minha
VOLTAS = 3        # voltas por largada na coleta. Escolha minha
torch.manual_seed(0)


def features(dist):
    """Scan → entrada da rede: poucos raios, cortados e escalados pra ficar entre 0 e 1."""
    return np.minimum(dist[::A_CADA], CORTE) / CORTE


def dirigir(politica, x, y, th, segundos, segs, caminho):
    """Roda o carro com uma política (estado, scan) → δ. Devolve trajetória, pares (scan, δ) e erros."""
    traj, X, Y, erros = [], [], [], []
    delta = 0.0
    for i in range(int(segundos / DT)):
        if i % PERIODO == 0:
            _, dist = escanear(x, y, th, segs)
            if dist.min() < CARRO:
                return np.array(traj), X, Y, np.array(erros), True
            delta = politica(x, y, th, dist)
            X.append(features(dist))
            Y.append(delta)
        x, y, th = passo(x, y, th, V, delta)
        traj.append((x, y))
        erros.append(cenoura(caminho, x, y, LD)[1])
    return np.array(traj), X, Y, np.array(erros), False


def especialista(caminho):
    def pol(x, y, th, dist):
        alvo, _ = cenoura(caminho, x, y, LD)
        return float(pure_pursuit(x, y, th, alvo, LD))
    return pol


def rede_politica(modelo):
    def pol(x, y, th, dist):  # repare: x, y, th chegam mas a rede NÃO usa
        with torch.no_grad():
            return float(modelo(torch.tensor(features(dist), dtype=torch.float32)))
    return pol


class Motorista(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.camadas = nn.Sequential(nn.Linear(n, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return torch.tanh(self.camadas(x)).squeeze(-1) * DELTA_MAX  # saída sempre dentro do batente


if __name__ == "__main__":
    segs, caminho = paredes(), pista_oval()
    volta_s = (2 * RETA + 2 * np.pi * RAIO) / V

    # 1. COLETAR: largadas um pouco diferentes, pra rede ver o carro fora do centro também
    largadas = [(0.0, y0, th0) for y0 in (-0.3, 0.0, 0.3) for th0 in (-0.15, 0.0, 0.15)]
    X, Y = [], []
    for x0, y0, th0 in largadas:
        _, xs, ys, _, _ = dirigir(especialista(caminho), x0, y0, th0, VOLTAS * volta_s, segs, caminho)
        X += xs
        Y += ys
    X, Y = torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    print(f"coleta: {len(Y)} pares (scan, volante) de {len(largadas)} largadas × {VOLTAS} voltas")
    espelhar = "--espelhar" in sys.argv
    if espelhar:
        # os raios vão de −135° a +135°, simétricos: inverter a ordem = trocar esquerda e direita
        X, Y = torch.cat([X, X.flip(1)]), torch.cat([Y, -Y])
        print(f"espelhado: {len(Y)} pares (metade agora vira pra direita)")

    # 2. TREINAR: separa 20% pra checar se aprendeu ou só decorou
    idx = torch.randperm(len(Y))
    n_val = len(Y) // 5
    val, tr = idx[:n_val], idx[n_val:]
    modelo = Motorista(X.shape[1])
    otim = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    for epoca in range(1500):
        otim.zero_grad()
        perda = ((modelo(X[tr]) - Y[tr]) ** 2).mean()
        perda.backward()
        otim.step()
    with torch.no_grad():
        erro_val = (modelo(X[val]) - Y[val]).abs().mean()
    print(f"treino: erro médio no volante (dados que a rede não viu) = {np.degrees(float(erro_val)):.2f}°")

    # 3. TESTAR: especialista × rede, da mesma largada; e uma largada que a rede nunca viu
    testes = [("largada normal", (0.0, 0.5, 0.0)), ("largada torta (30°)", (0.0, 0.0, 0.52))]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (nome, (x0, y0, th0)) in zip(axs, testes):
        for s in segs[::4]:
            ax.plot(s[:, 0], s[:, 1], "k-", lw=0.6)
        for quem, pol in (("especialista", especialista(caminho)), ("rede", rede_politica(modelo))):
            traj, _, _, erros, bateu = dirigir(pol, x0, y0, th0, 2 * volta_s, segs, caminho)
            e = erros[len(erros) // 5:] if len(erros) > 10 else erros
            status = "BATEU" if bateu else f"erro médio {e.mean() * 100:.1f} cm, máx {e.max() * 100:.1f} cm"
            print(f"{nome:>20} | {quem:>12}: {status}")
            if len(traj):
                ax.plot(traj[:, 0], traj[:, 1], lw=1.5, label=f"{quem}" + (" (BATEU)" if bateu else ""))
        ax.plot(x0, y0, "ko")
        ax.set_title(nome)
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    sufixo = "_espelhado" if espelhar else ""
    fig.savefig(out / f"imitacao{sufixo}.png", dpi=120)
    torch.save(modelo.state_dict(), out / f"motorista{sufixo}.pt")
    print(f"gráfico: {out / f'imitacao{sufixo}.png'} | modelo: {out / f'motorista{sufixo}.pt'}")
