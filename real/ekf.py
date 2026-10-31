"""Filtro de Kalman estendido (EKF) pra pose 2D, passo a passo (online). Usado no KITTI e no CARLA.

Estado: [x, y, θ] ou, com estimar_vies=True, [x, y, θ, b], onde b é o viés do giroscópio (rad/s):
um erro fixo que o giroscópio soma sempre pro mesmo lado. Sem b no estado, o filtro acha que o erro
do giro é ruído que some na média; como não some, ele fica confiante demais e atrasado.

- prever(v, giro, dt): anda com o modelo de bicicleta e aumenta a incerteza P.
- corrigir(gx, gy): puxa a estimativa pro GPS na medida do ganho K = P Hᵀ (H P Hᵀ + R)⁻¹.
  O GPS não mede b, mas b aparece aos poucos: se o GPS sempre "puxa" a curva pro mesmo lado,
  a correlação em P entre θ e b empurra b na direção certa.
"""
import numpy as np


class EKF:
    def __init__(self, x, y, th, sigma_gps, sigma_v, sigma_giro, estimar_vies=False,
                 vies_inicial=np.radians(1.0), deriva_vies=np.radians(0.01)):
        # vies_inicial: até ~1°/s de viés antes de saber qualquer coisa (chute, giroscópio de celular)
        # deriva_vies: o viés muda devagar, ~0,01°/s por √s (chute: random walk)
        self.n = 4 if estimar_vies else 3
        self.s = np.array([x, y, th, 0.0][: self.n], float)
        self.P = np.diag([0.1, 0.1, np.radians(1), vies_inicial][: self.n]) ** 2  # largada quase certa
        self.R = np.eye(2) * sigma_gps**2
        self.Qu = np.diag([sigma_v**2, sigma_giro**2])
        self.deriva = deriva_vies
        self.H = np.zeros((2, self.n))
        self.H[0, 0] = self.H[1, 1] = 1.0  # o GPS mede x e y

    def prever(self, v, giro, dt):
        th = self.s[2]
        b = self.s[3] if self.n == 4 else 0.0
        self.s[0] += v * np.cos(th) * dt
        self.s[1] += v * np.sin(th) * dt
        self.s[2] += (giro - b) * dt
        F = np.eye(self.n)
        F[0, 2], F[1, 2] = -v * np.sin(th) * dt, v * np.cos(th) * dt
        G = np.zeros((self.n, 2))
        G[0, 0], G[1, 0], G[2, 1] = np.cos(th) * dt, np.sin(th) * dt, dt
        if self.n == 4:
            F[2, 3] = -dt  # mais viés estimado → menos giro
        self.P = F @ self.P @ F.T + G @ self.Qu @ G.T
        if self.n == 4:
            self.P[3, 3] += self.deriva**2 * dt

    def corrigir(self, gx, gy):
        inov = np.array([gx, gy]) - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ inov
        self.P = (np.eye(self.n) - K @ self.H) @ self.P


if __name__ == "__main__":
    # checagem: giroscópio com viés de 0,5°/s andando em linha reta; GPS perfeito com ruído pequeno.
    # O filtro com viés no estado tem que descobrir ~0,5°/s sozinho.
    r = np.random.default_rng(0)
    f = EKF(0, 0, 0, sigma_gps=1.0, sigma_v=0.1, sigma_giro=np.radians(0.5), estimar_vies=True)
    for k in range(1, 601):  # 60 s a 10 Hz, 10 m/s em linha reta
        f.prever(10.0, np.radians(0.5) + r.normal(0, np.radians(0.5)), 0.1)
        if k % 10 == 0:
            f.corrigir(10.0 * k * 0.1 + r.normal(0, 1.0), r.normal(0, 1.0))
    print(f"viés estimado {np.degrees(f.s[3]):.2f}°/s (verdade 0,50)")
    assert abs(np.degrees(f.s[3]) - 0.5) < 0.15
