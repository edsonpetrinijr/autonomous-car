"""Dado real, passos 2 e 3: onde o carro está, sem olhar a resposta.

Passo 2, dead reckoning ("andar às cegas"): parte da pose certa e, a cada quadro, integra só a velocidade
e o giro (a bicicleta do degrau 1, agora com sensor de verdade). Nunca olha o GPS.

Passo 3, EKF (filtro de Kalman estendido): o "prever-corrigir" do degrau 5 com dado real.
- prever: anda com velocidade e giro (suave, mas deriva) e aumenta a incerteza P;
- corrigir: quando chega um GPS ruim (pula, mas não deriva), puxa a estimativa pra ele
  na medida certa: K = P / (P + R). Muita incerteza minha → confio no GPS; GPS muito ruim → confio em mim.

Uso: .venv/bin/python real/estimativa.py [drive]
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kitti import DRIVE, verdade

rng = np.random.default_rng(0)


def dead_reckoning(t, v, giro, x0, y0, yaw0):
    """Integra v (m/s, pra frente) e giro (rad/s) com Euler, usando o dt real entre quadros."""
    x, y, th = [x0], [y0], [yaw0]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        x.append(x[-1] + v[k - 1] * np.cos(th[-1]) * dt)
        y.append(y[-1] + v[k - 1] * np.sin(th[-1]) * dt)
        th.append(th[-1] + giro[k - 1] * dt)
    return np.array(x), np.array(y), np.array(th)


SIGMA_GPS = 3.0          # m: GPS de celular, ordem de grandeza (chute). O KITTI tem RTK de ~0,4 m: estragamos de propósito
GPS_A_CADA = 10          # quadros: GPS a 1 Hz (o LiDAR/IMU vem a 10 Hz). Típico de receptor comum
SIGMA_V = 0.2            # m/s: o quanto confio na velocidade a cada passo (chute; ver sensor comum)
SIGMA_GIRO = np.radians(1.0)  # rad/s: idem pro giroscópio


def ekf(t, v, giro, gps, x0, y0, yaw0):
    """Estado [x, y, θ]. gps: (N, 2) com NaN nos quadros sem leitura. Devolve a trajetória estimada."""
    s = np.array([x0, y0, yaw0], float)
    P = np.diag([0.1, 0.1, np.radians(1)]) ** 2  # começo quase certo (sabemos a largada)
    R = np.eye(2) * SIGMA_GPS**2
    H = np.array([[1.0, 0, 0], [0, 1.0, 0]])     # o GPS mede x e y, não θ
    saida = [s.copy()]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        th = s[2]
        # prever: mesmo modelo do dead reckoning
        s = s + np.array([v[k - 1] * np.cos(th) * dt, v[k - 1] * np.sin(th) * dt, giro[k - 1] * dt])
        F = np.array([[1, 0, -v[k - 1] * np.sin(th) * dt], [0, 1, v[k - 1] * np.cos(th) * dt], [0, 0, 1]])
        G = np.array([[np.cos(th) * dt, 0], [np.sin(th) * dt, 0], [0, dt]])  # como erro em v e giro vira erro de pose
        P = F @ P @ F.T + G @ np.diag([SIGMA_V**2, SIGMA_GIRO**2]) @ G.T
        # corrigir: só quando tem GPS
        if not np.isnan(gps[k, 0]):
            inov = gps[k] - H @ s                        # quanto o GPS discorda de mim
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)  # ganho de Kalman
            s = s + K @ inov
            P = (np.eye(3) - K @ H) @ P
        saida.append(s.copy())
    return np.array(saida)


def gps_ruim(x, y, semente):
    r = np.random.default_rng(semente)
    g = np.full((len(x), 2), np.nan)
    i = np.arange(0, len(x), GPS_A_CADA)
    g[i] = np.c_[x[i], y[i]] + r.normal(0, SIGMA_GPS, (len(i), 2))
    return g


def rmse(xe, ye, x, y):
    return np.sqrt(np.mean((xe - x) ** 2 + (ye - y) ** 2))


def erro(xe, ye, x, y):
    e = np.hypot(xe - x, ye - y)
    return e[-1], e.max()


if __name__ == "__main__":
    t, x, y, yaw, v, giro = verdade()
    dist = np.sum(np.hypot(np.diff(x), np.diff(y)))

    # sensor de laboratório: o próprio OXTS (IMU de fibra óptica + velocidade do GPS)
    xb, yb, _ = dead_reckoning(t, v, giro, x[0], y[0], yaw[0])
    # sensor de carro comum (chute): roda com 2% de erro de escala, giroscópio com viés de 0,5°/s e ruído
    v_barato = v * 1.02 + rng.normal(0, 0.1, len(v))
    giro_barato = giro + np.radians(0.5) + rng.normal(0, np.radians(1), len(giro))
    xc, yc, _ = dead_reckoning(t, v_barato, giro_barato, x[0], y[0], yaw[0])

    print(f"{DRIVE}: {dist:.0f} m em {t[-1]:.1f} s")
    for nome, xe, ye in (("IMU de laboratório", xb, yb), ("sensor de carro comum (chute)", xc, yc)):
        fim, mx = erro(xe, ye, x, y)
        print(f"  {nome:>30}: erro no fim {fim:.2f} m ({fim / dist * 100:.1f}% da distância), máximo {mx:.2f} m")

    # passo 3: 20 sorteios de ruído do GPS (uma execução só não prova nada)
    print(f"\nGPS ruim: σ = {SIGMA_GPS} m a {10 / GPS_A_CADA:.0f} Hz | erro médio (RMSE), média de 20 sorteios")
    res = {"GPS cru (só nos instantes com leitura)": [], "EKF, IMU de laboratório": [], "EKF, sensor comum (chute)": []}
    for sem in range(20):
        g = gps_ruim(x, y, sem)
        tem = ~np.isnan(g[:, 0])
        res["GPS cru (só nos instantes com leitura)"].append(rmse(g[tem, 0], g[tem, 1], x[tem], y[tem]))
        e = ekf(t, v, giro, g, x[0], y[0], yaw[0])
        res["EKF, IMU de laboratório"].append(rmse(e[:, 0], e[:, 1], x, y))
        e2 = ekf(t, v_barato, giro_barato, g, x[0], y[0], yaw[0])
        res["EKF, sensor comum (chute)"].append(rmse(e2[:, 0], e2[:, 1], x, y))
    for nome, r in res.items():
        print(f"  {nome:>38}: {np.mean(r):.2f} m  (de {np.min(r):.2f} a {np.max(r):.2f})")
    print(f"  {'às cegas, IMU de laboratório':>38}: {rmse(xb, yb, x, y):.2f} m")
    print(f"  {'às cegas, sensor comum (chute)':>38}: {rmse(xc, yc, x, y):.2f} m")
    g0 = gps_ruim(x, y, 0)
    e0 = ekf(t, v_barato, giro_barato, g0, x[0], y[0], yaw[0])

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(g0[:, 0], g0[:, 1], "gx", ms=8, label=f"GPS ruim (σ {SIGMA_GPS:g} m, 1 Hz)")
    ax.plot(e0[:, 0], e0[:, 1], "g-", lw=2, label="EKF (sensor comum + GPS ruim)")
    ax.plot(x, y, "k-", lw=3, label="verdade (GPS/INS RTK)")
    ax.plot(xb, yb, "b--", lw=2, label="às cegas, IMU de laboratório")
    ax.plot(xc, yc, "r--", lw=2, label="às cegas, sensor comum (chute)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"KITTI ({dist:.0f} m): às cegas × filtro de Kalman")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{DRIVE}_dead_reckoning.png", dpi=120, bbox_inches="tight")
    print(f"gráfico: {out / f'{DRIVE}_dead_reckoning.png'}")

    # o gráfico que conta a história: erro ao longo do tempo (sorteio 0 do GPS)
    e_lab = ekf(t, v, giro, g0, x[0], y[0], yaw[0])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tem = ~np.isnan(g0[:, 0])
    ax.plot(t[tem], np.hypot(g0[tem, 0] - x[tem], g0[tem, 1] - y[tem]), "gx", label="GPS ruim sozinho")
    for nome, xe, ye, est in (("às cegas, sensor comum", xc, yc, "r--"), ("às cegas, IMU de laboratório", xb, yb, "b--"),
                               ("filtro, sensor comum + GPS", e0[:, 0], e0[:, 1], "r-"),
                               ("filtro, IMU de laboratório + GPS", e_lab[:, 0], e_lab[:, 1], "b-")):
        ax.plot(t, np.hypot(xe - x, ye - y), est, lw=2, label=nome)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("erro de posição (m)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"KITTI {DRIVE}: às cegas o erro cresce; com GPS ele fica limitado")
    fig.savefig(out / f"{DRIVE}_erro_tempo.png", dpi=120, bbox_inches="tight")
    print(f"gráfico: {out / f'{DRIVE}_erro_tempo.png'}")
