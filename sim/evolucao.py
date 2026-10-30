"""Meu modelo, passo 2: evolução (estilo Deep Cars, do Universo Programado). SEM especialista.

1. Cria uma população de carros, cada um com uma rede de pesos ALEATÓRIOS.
   Como no Deep Cars, a rede controla volante E velocidade (0,5 a 8 m/s).
2. Todos dirigem ao mesmo tempo por T segundos, só com o LiDAR.
   Nota (fitness) = quantos metros andou na pista antes de bater. Mais rápido = mais metros,
   mas o pneu tem limite (degrau 4): rápido demais na curva, escorrega e bate.
3. Os melhores sobrevivem; os filhos são cópias deles com pequenas mutações aleatórias nos pesos.
4. Repete. Ninguém ensina a dirigir: a seleção descobre sozinha.

Pra rodar rápido no Mac, simula a população inteira de uma vez (numpy) e usa só 16 raios.

Uso: .venv/bin/python sim/evolucao.py            (nota = metros)
     .venv/bin/python sim/evolucao.py --limpo    (punição 5 por raspar na parede)
     .venv/bin/python sim/evolucao.py --punicao N  (outro peso)
     --semente S  (outro sorteio inicial; vale com qualquer das opções acima)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bicicleta import L
from lidar import CARRO, FOV, MEIA_PISTA
from pista_nova import pista_feijao
from pure_pursuit import Y0, pista_oval
from reta import DELTA_MAX
from velocidade import A_MAX

V_MIN, V_MAX = 0.5, 8.0  # m/s: a rede escolhe entre isso. 8 = mesmo teto do perfil do degrau 4
T = 15.0            # s de corrida por geração. Escolha minha
DT = 0.01           # s
A_CADA = 5          # a rede decide a cada 5 passos = 20 Hz. Escolha minha (rápido o bastante a 2 m/s)
N_RAIOS = 16        # Deep Cars usa 18. Menos raios = mais rápido
CORTE = 5.0         # m, igual à imitação
OCULTOS = 8         # neurônios escondidos. Deep Cars usa 6. Rede pequena evolui mais fácil
POP = 200           # carros por geração. Deep Cars usa 1000; 200 cabe no Mac
ELITE = 20          # os 10% melhores viram pais
SIGMA = 0.1         # tamanho da mutação. Chute
GERACOES = 30
MARGEM = 0.15       # m de folga até a parede abaixo disso = "raspando". Chute: meia largura do carro
PUNICAO = 5.0       # m perdidos por segundo raspando. Chute: ~velocidade média → raspar vale o mesmo que ficar parado
LIMPO = "--limpo" in sys.argv or "--punicao" in sys.argv
if "--punicao" in sys.argv:  # --punicao N: testa outro peso (o --limpo sozinho usa 5)
    PUNICAO = float(sys.argv[sys.argv.index("--punicao") + 1])
SEMENTE = int(sys.argv[sys.argv.index("--semente") + 1]) if "--semente" in sys.argv else 0  # sorteio inicial da população
SUFIXO = ("_limpo" if PUNICAO == 5.0 else f"_p{PUNICAO:g}") if LIMPO else ""
SUFIXO += f"_s{SEMENTE}" if SEMENTE else ""
rng = np.random.default_rng(SEMENTE)

SAIDAS = 2          # volante e velocidade
N_PARAMS = N_RAIOS * OCULTOS + OCULTOS + OCULTOS * SAIDAS + SAIDAS  # pesos e vieses das duas camadas


def segmentos(caminho, passo=4):
    """Paredes a ±MEIA_PISTA da linha central, com 1 ponto a cada `passo` (menos segmentos = mais rápido)."""
    c = caminho[::passo]
    t = np.roll(c, -1, axis=0) - c
    t /= np.hypot(t[:, 0], t[:, 1])[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    segs = []
    for lado in (+1, -1):
        p = c + lado * MEIA_PISTA * n
        segs.append(np.stack([p, np.roll(p, -1, axis=0)], axis=1))
    return np.concatenate(segs)


def escanear(pos, th, segs):
    """LiDAR de N carros ao mesmo tempo. pos (N,2), th (N,) → distâncias (N, N_RAIOS)."""
    angs = np.linspace(-FOV / 2, FOV / 2, N_RAIOS)
    a = th[:, None] + angs[None]                                     # (N,R)
    d = np.stack([np.cos(a), np.sin(a)], axis=-1)[:, :, None, :]     # (N,R,1,2)
    s0, e = segs[:, 0], segs[:, 1] - segs[:, 0]                      # (S,2)
    ap = (s0[None] - pos[:, None])[:, None]                          # (N,1,S,2)
    cr = lambda p, q: p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
    den = cr(d, e[None, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cr(ap, e[None, None]) / den
        u = cr(ap, d) / den
    ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    return np.where(ok, t, np.inf).min(axis=2)


def rede(params, x):
    """MLP de N carros: entrada (N, N_RAIOS) → (volante, velocidade), cada um (N,). Cada carro com os seus pesos."""
    i = 0
    W1 = params[:, i:i + N_RAIOS * OCULTOS].reshape(-1, N_RAIOS, OCULTOS); i += N_RAIOS * OCULTOS
    b1 = params[:, i:i + OCULTOS]; i += OCULTOS
    W2 = params[:, i:i + OCULTOS * SAIDAS].reshape(-1, OCULTOS, SAIDAS); i += OCULTOS * SAIDAS
    b2 = params[:, i:i + SAIDAS]
    h = np.tanh(np.einsum("nr,nro->no", x, W1) + b1)
    s = np.tanh(np.einsum("no,nos->ns", h, W2) + b2)  # duas saídas entre −1 e 1
    return s[:, 0] * DELTA_MAX, V_MIN + (s[:, 1] + 1) / 2 * (V_MAX - V_MIN)


def simular(params, caminho, segs, segundos, largada, guardar=False, punir=0.0):
    """Todos os carros dirigem juntos. Devolve metros andados, se bateu, erro lateral médio e trajetórias."""
    n = len(params)
    pos = np.tile(np.array(largada[:2], dtype=float), (n, 1))
    th = np.full(n, float(largada[2]))
    vivo = np.ones(n, bool)
    idx = np.full(n, int(np.argmin(np.hypot(*(caminho - pos[0]).T))))  # ponto da linha central mais perto da largada
    progresso = np.zeros(n)
    erro_soma, erro_n = np.zeros(n), np.zeros(n)
    trajs, vivos = ([pos.copy()], [vivo.copy()]) if guardar else (None, None)
    m = len(caminho)
    delta, v = np.zeros(n), np.zeros(n)
    for passo_i in range(int(segundos / DT)):
        if passo_i % A_CADA == 0:
            dist = escanear(pos, th, segs)
            vivo &= dist.min(axis=1) >= CARRO
            raspando = vivo & (dist.min(axis=1) - CARRO < MARGEM)
            progresso -= punir * raspando * DT * A_CADA  # só o treino usa (punir > 0)
            delta, v = rede(params, np.minimum(dist, CORTE) / CORTE)
            # limite do pneu (degrau 4): a curva que dá pra fazer nessa velocidade. Trapaça: v muda na hora
            k = np.tan(delta) / L
            k_max = A_MAX / v**2
            delta = np.arctan(np.clip(k, -k_max, k_max) * L)
            # progresso ao longo da linha central (o carro NÃO vê isso; é só a nota)
            novo = np.argmin(np.hypot(caminho[None, :, 0] - pos[:, None, 0], caminho[None, :, 1] - pos[:, None, 1]), axis=1)
            avanco = (novo - idx + m // 2) % m - m // 2
            progresso += np.where(vivo, avanco, 0) * 0.05
            idx = novo
            e = np.hypot(*(caminho[novo] - pos).T)
            erro_soma += np.where(vivo, e, 0)
            erro_n += vivo
        mv = vivo.astype(float)  # carro batido fica parado
        pos[:, 0] += mv * v * np.cos(th) * DT
        pos[:, 1] += mv * v * np.sin(th) * DT
        th += mv * v / L * np.tan(delta) * DT
        if guardar and passo_i % A_CADA == 0:
            trajs.append(pos.copy())
            vivos.append(vivo.copy())
    if guardar:
        return progresso, ~vivo, erro_soma / np.maximum(erro_n, 1), np.array(trajs), np.array(vivos)
    return progresso, ~vivo, erro_soma / np.maximum(erro_n, 1), None


def folga(traj, caminho):
    """Distância da lateral do carro até a parede mais próxima, em cada ponto da trajetória (T,2)."""
    lateral = np.hypot(traj[:, None, 0] - caminho[None, :, 0], traj[:, None, 1] - caminho[None, :, 1]).min(axis=1)
    return MEIA_PISTA - lateral - CARRO


if __name__ == "__main__":
    oval = pista_oval()
    segs_oval = segmentos(oval)
    volta = len(oval) * 0.05
    largada = (0.0, Y0, 0.0)

    pop = rng.normal(0, 0.5, (POP, N_PARAMS))
    hist_melhor, hist_media, campeoes, populacoes = [], [], {}, {}
    print(f"referência: especialista com perfil de velocidade (degrau 4) = 4,26 s por volta "
          f"→ ~{T / 4.26 * volta:.0f} m em {T:.0f} s")
    for g in range(1, GERACOES + 1):
        nota, bateu, _, _ = simular(pop, oval, segs_oval, T, largada, punir=PUNICAO if LIMPO else 0.0)
        ordem = np.argsort(-nota)
        hist_melhor.append(nota[ordem[0]])
        hist_media.append(nota.mean())
        if g in (1, 5, 15, GERACOES):
            campeoes[g] = pop[ordem[0]].copy()
        if g in (1, 2, 5, GERACOES):
            populacoes[f"g{g}"] = pop[ordem].copy()  # ordenada: o índice 0 é o campeão dessa geração
        b = nota[ordem[0]]
        print(f"geração {g:2d}: melhor andou {b:5.1f} m ({b / volta:.1f} voltas, "
              f"{T * volta / b if b > 0 else 0:4.2f} s/volta) | média {nota.mean():5.1f} m | bateram {bateu.sum():3d}")
        pais = pop[ordem[:ELITE]]
        filhos = pais[rng.integers(0, ELITE, POP - ELITE)] + rng.normal(0, SIGMA, (POP - ELITE, N_PARAMS))
        pop = np.vstack([pais, filhos])  # os pais continuam (elitismo): o melhor nunca se perde

    melhor = campeoes[GERACOES][None]
    np.save(Path(__file__).parent / "out" / f"evolucao_melhor{SUFIXO}.npy", melhor)
    np.savez(Path(__file__).parent / "out" / f"evolucao_populacoes{SUFIXO}.npz", **populacoes)  # pro vídeo
    feijao = pista_feijao()
    d0 = feijao[1] - feijao[0]
    print()
    for nome, cam, larg in (("oval (treino)", oval, largada),
                            ("feijão (nunca viu)", feijao, (*feijao[0], np.arctan2(d0[1], d0[0])))):
        nota, bateu, erro, _ = simular(melhor, cam, segmentos(cam), T, larg)
        vc = len(cam) * 0.05
        status = (f"BATEU depois de {nota[0]:.1f} m" if bateu[0]
                  else f"{nota[0] / vc:.1f} voltas em {T:.0f} s ({T * vc / nota[0]:.2f} s/volta), erro lateral médio {erro[0] * 100:.1f} cm")
        _, _, _, tr, _ = simular(melhor, cam, segmentos(cam), T, larg, guardar=True)
        f = folga(tr[:, 0], cam)
        print(f"campeão no {nome:>19}: {status}")
        print(f"{'':>31}folga mínima até a parede {f.min() * 100:.0f} cm, raspando {(f < MARGEM).mean() * 100:.0f}% do tempo")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(range(1, GERACOES + 1), hist_melhor, label="melhor carro")
    axs[0].plot(range(1, GERACOES + 1), hist_media, label="média da população")
    axs[0].axhline(T / 4.26 * volta, color="k", ls="--", lw=1, label="especialista (perfil, degrau 4)")
    axs[0].set_xlabel("geração")
    axs[0].set_ylabel("metros andados antes de bater")
    axs[0].legend()
    axs[0].grid(True)
    for s in segs_oval:
        axs[1].plot(s[:, 0], s[:, 1], "k-", lw=0.6)
    for g, p in campeoes.items():
        _, _, _, tr, _ = simular(p[None], oval, segs_oval, T, largada, guardar=True)
        axs[1].plot(tr[:, 0, 0], tr[:, 0, 1], lw=1.5, label=f"campeão da geração {g}")
    axs[1].axis("equal")
    axs[1].grid(True)
    axs[1].legend(fontsize=8)
    axs[1].set_title("Melhor carro de cada geração")
    fig.tight_layout()
    png = Path(__file__).parent / "out" / f"evolucao{SUFIXO}.png"
    fig.savefig(png, dpi=120)
    print(f"gráfico: {png}")
