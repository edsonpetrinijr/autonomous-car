"""Roda a evolução com vários pesos de punição e várias sementes, e resume numa tabela.

Uso: .venv/bin/python sim/varredura.py   (~10 min; usa 5 processos em paralelo)
"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PESOS = [0, 1, 5, 20]
SEMENTES = [0, 1, 2]
aqui = Path(__file__).parent


def rodar(job):
    w, s = job
    cmd = [sys.executable, str(aqui / "evolucao.py"), "--semente", str(s)] + (["--punicao", str(w)] if w else [])
    return job, subprocess.run(cmd, capture_output=True, text=True).stdout


def ler(saida):
    """Do texto do evolucao.py: (s/volta no oval, folga mínima cm, % raspando, feijão ok?)"""
    oval = re.search(r"oval \(treino\): (.*)", saida).group(1)
    volta = float(re.search(r"\(([\d.]+) s/volta", oval).group(1)) if "BATEU" not in oval else np.nan
    folga, rasp = (int(x) for x in re.search(r"folga mínima até a parede (-?\d+) cm, raspando (\d+)%", saida).groups())
    feijao = "BATEU" not in re.search(r"feijão \(nunca viu\): (.*)", saida).group(1)
    return volta, folga, rasp, feijao


if __name__ == "__main__":
    jobs = [(w, s) for w in PESOS for s in SEMENTES]
    with ThreadPoolExecutor(5) as ex:
        res = dict(ex.map(rodar, jobs))
    (aqui / "out" / "varredura.txt").write_text("\n\n".join(f"== peso {w} semente {s}\n{o}" for (w, s), o in res.items()))
    print(f"{'peso':>5} | {'s/volta (3 sementes)':>24} | {'folga mín (cm)':>16} | {'raspando %':>12} | feijão sem bater")
    for w in PESOS:
        r = [ler(res[(w, s)]) for s in SEMENTES]
        fmt = lambda i: " ".join(f"{x[i]:>5.2f}" if i == 0 else f"{x[i]:>4d}" for x in r)
        print(f"{w:>5} | {fmt(0):>24} | {fmt(1):>16} | {fmt(2):>12} | {sum(x[3] for x in r)} de {len(r)}")
