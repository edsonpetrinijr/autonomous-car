"""Dado real, passo 1: ler um trecho do KITTI e desenhar o caminho do carro com o LiDAR por cima.

KITTI raw (Geiger et al., 2013): carro de verdade em Karlsruhe, 2011. Por quadro (10 Hz):
- OXTS: GPS/INS de alta precisão (RTK). Serve de VERDADE pra medir o erro dos nossos filtros.
- Velodyne HDL-64: nuvem de pontos 3D (x, y, z, refletância) no referencial do LiDAR.

Uso: .venv/bin/python real/kitti.py [drive]   (padrão: 2011_09_26_drive_0005)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent / "data" / "kitti"
DRIVE = sys.argv[1] if len(sys.argv) > 1 else "2011_09_26_drive_0005"
DIA = DRIVE[:10]
PASTA = RAIZ / DIA / f"{DRIVE}_sync"
RAIO_TERRA = 6378137.0  # m, WGS84 (mesma conta do devkit do KITTI)


def ler_oxts():
    """Devolve t (s, desde o 1º quadro) e uma matriz (N, 30) com as colunas do dataformat.txt."""
    arqs = sorted((PASTA / "oxts" / "data").glob("*.txt"))
    dados = np.array([np.loadtxt(a) for a in arqs])
    ts = np.array([np.datetime64(l.strip()[:26]) for l in open(PASTA / "oxts" / "timestamps.txt")])
    t = (ts - ts[0]) / np.timedelta64(1, "s")
    return t.astype(float), dados


def lat_lon_para_xy(lat, lon):
    """Projeção de Mercator com escala fixa no 1º ponto (devkit do KITTI). Origem no 1º quadro, metros."""
    s = np.cos(np.radians(lat[0]))
    x = s * np.radians(lon) * RAIO_TERRA
    y = s * RAIO_TERRA * np.log(np.tan(np.radians(90 + lat) / 2))
    return x - x[0], y - y[0]


def verdade():
    """Pose 2D de referência: t, x, y (m), yaw (rad, 0 = leste), v (m/s, pra frente), giro (rad/s, em torno de z)."""
    t, d = ler_oxts()
    x, y = lat_lon_para_xy(d[:, 0], d[:, 1])
    return t, x, y, d[:, 5], d[:, 8], d[:, 22]  # yaw, vf, wu (giro em torno do "pra cima" do carro)


def imu_para_velo():
    """Calibração: ponto no LiDAR → ponto no referencial da IMU (inversa do arquivo, que vai IMU → LiDAR)."""
    lin = {l.split(":")[0]: l.split(":")[1] for l in open(RAIZ / DIA / "calib_imu_to_velo.txt")}
    R = np.array(lin["R"].split(), float).reshape(3, 3)
    T = np.array(lin["T"].split(), float)
    return R.T, -R.T @ T


def lidar(i):
    """Quadro i do Velodyne: (N, 4) = x, y, z, refletância, em metros no referencial do LiDAR."""
    return np.fromfile(PASTA / "velodyne_points" / "data" / f"{i:010d}.bin", dtype=np.float32).reshape(-1, 4)


def fatia_no_mundo(i, x, y, yaw, z_min=-1.3, z_max=0.5, passo=4):
    """Pontos do LiDAR numa fatia de altura (sem chão, sem copa de árvore), levados pro mundo 2D.
    O LiDAR fica a ~1,73 m do chão (paper do KITTI), então z = −1,3 m ≈ 40 cm acima do asfalto. Corte = escolha minha.
    Ignora roll e pitch (carro quase plano): erro pequeno pra um desenho de cima."""
    p = lidar(i)[::passo]
    p = p[(p[:, 2] > z_min) & (p[:, 2] < z_max), :3]
    R, T = imu_para_velo()
    q = p @ R.T + T                       # LiDAR → IMU
    c, s = np.cos(yaw[i]), np.sin(yaw[i])
    return np.c_[x[i] + c * q[:, 0] - s * q[:, 1], y[i] + s * q[:, 0] + c * q[:, 1]]


if __name__ == "__main__":
    t, x, y, yaw, v, giro = verdade()
    dist = np.sum(np.hypot(np.diff(x), np.diff(y)))
    print(f"{DRIVE}: {len(t)} quadros em {t[-1]:.1f} s, {dist:.0f} m percorridos, "
          f"v média {v.mean():.1f} m/s ({v.mean() * 3.6:.0f} km/h), giro total {np.degrees(yaw[-1] - yaw[0]):.0f}°")
    _, d = ler_oxts()
    print(f"precisão de posição informada pelo OXTS: {d[:, 23].min():.2f} a {d[:, 23].max():.2f} m")

    fig, ax = plt.subplots(figsize=(9, 9))
    tem = [i for i in range(0, len(t), 5) if (PASTA / "velodyne_points" / "data" / f"{i:010d}.bin").exists()]  # o 0009 tem buracos no LiDAR
    pts = np.vstack([fatia_no_mundo(i, x, y, yaw) for i in tem])
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c="gray", alpha=0.4, label="LiDAR (fatia de 40 cm a 2,2 m de altura)")
    ax.plot(x, y, "r-", lw=2, label="caminho real (GPS/INS RTK)")
    ax.plot(x[0], y[0], "go", ms=8, label="início")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"KITTI {DRIVE}: {dist:.0f} m em {t[-1]:.0f} s")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{DRIVE}_mapa.png", dpi=120, bbox_inches="tight")
    print(f"gráfico: {out / f'{DRIVE}_mapa.png'}")
