"""As ruas de verdade do KITTI no CARLA, e o meu pure pursuit dirigindo a rota que o carro do KITTI fez.

1. Converte o pedaço do OpenStreetMap de Karlsruhe (mapas/karlsruhe_kitti.osm) em OpenDRIVE e carrega no CARLA.
   Só ruas: o conversor do CARLA não traz prédios. Ruas acabam secas na borda do mapa.
2. Leva a rota do KITTI (lat/lon, mapas/kitti_0009_rota.csv) pras coordenadas do CARLA e desenha no chão.
   Teste: distância da rota até o centro da faixa mais próxima. Se o alinhamento estiver certo, uns poucos metros.
3. --dirigir: o carro do CARLA segue essa rota com pure pursuit (degrau 3) na velocidade que o KITTI andou.
   Compara o volante do meu controlador com o volante que o motorista humano deve ter usado:
   δ_humano = atan(L · giro / v) (bicicleta, degrau 1).

Mapa: © OpenStreetMap contributors (ODbL). Rota: KITTI raw (Geiger et al., 2013), CC BY-NC-SA 3.0.

Como rodar (máquina Windows do lab, CARLA 0.9.16 aberto, Python 3.10):
  py -3.10 osm_kitti.py                       # carrega o mapa e desenha a rota (olhe no espectador)
  py -3.10 osm_kitti.py --dirigir             # + pure pursuit seguindo a rota
  py -3.10 osm_kitti.py --dirigir --video     # + mp4 com câmera atrás do carro (pip install imageio[ffmpeg])
  py -3.10 osm_kitti.py --dirigir --localizar # o carro NÃO sabe onde está: usa o EKF (real/ekf.py) com
                                              # GPS ruim + velocidade + giroscópio com viés, simulados
  py -3.10 osm_kitti.py --dirigir --localizar --sem-vies   # EKF sem o viés no estado, pra comparar
  py -3.10 osm_kitti.py --dirigir --localizar --escala-v 0.02   # + roda 2% errada, que o EKF não conhece

4. --localizar: em vez de perguntar a posição exata pro CARLA, o carro estima onde está com o mesmo
   EKF que foi testado no KITTI. Os sensores são simulados a partir do movimento real do carro:
   GPS com ruído de 3 m a 1 Hz, velocidade da roda com ruído, giroscópio com ruído e viés.
"""
import argparse
import io
import math
import queue
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

import numpy as np

try:
    import carla
except ImportError:  # sem CARLA dá pra importar as funções puras (pure_pursuit, dist_polilinha) e testar no Mac
    carla = None

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI.parent / "real"))
from ekf import EKF  # noqa: E402  (o mesmo filtro do KITTI)
DT = 0.05            # s, modo síncrono (igual ao circulo.py)
L_KITTI = 2.71       # m: entre-eixos do VW Passat B6 do KITTI. Ficha técnica de memória, VERIFICAR antes de citar
SIGMA_GPS, GPS_A_CADA = 3.0, 20   # m e passos: GPS de celular (chute) a 1 Hz, igual ao teste do KITTI
SIGMA_V, SIGMA_GIRO = 0.2, math.radians(1.0)  # ruído da roda e do giroscópio (chute, os mesmos do KITTI)
LD_MIN, LD_GANHO = 4.0, 0.8  # lookahead = max(4 m, 0,8 s × v). Chute; mesma lógica do degrau 3 (ld cresce com v)
TIPOS_DE_RUA = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"]


def recortar_osm(dados, lat, lon, margem):
    """Corta o .osm pra só as ruas a até `margem` metros da rota. O mapa inteiro (1,7 km, ~1000 ruas)
    derruba o CARLA na hora de gerar a malha 3D. Mantém a rua inteira se um nó dela cai na caixa."""
    raiz = ET.fromstring(dados)
    dlat = margem / 111320.0
    dlon = margem / (111320.0 * math.cos(math.radians(float(np.mean(lat)))))
    s, n = float(np.min(lat)) - dlat, float(np.max(lat)) + dlat
    o, l = float(np.min(lon)) - dlon, float(np.max(lon)) + dlon
    dentro = {nd.get("id") for nd in raiz.iter("node")
              if s <= float(nd.get("lat")) <= n and o <= float(nd.get("lon")) <= l}
    ruas = [w for w in raiz.iter("way") if any(r.get("ref") in dentro for r in w.iter("nd"))]
    usados = {r.get("ref") for w in ruas for r in w.iter("nd")}
    novo = ET.Element("osm", raiz.attrib)
    ET.SubElement(novo, "bounds", minlat=f"{s:.7f}", minlon=f"{o:.7f}", maxlat=f"{n:.7f}", maxlon=f"{l:.7f}")
    novo.extend(nd for nd in raiz.iter("node") if nd.get("id") in usados)
    novo.extend(ruas)
    print(f"OSM recortado: {len(ruas)} vias (margem de {margem:.0f} m em volta da rota)")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(novo, encoding="unicode")


def carregar_mapa(client, osm, lat=None, lon=None, margem=300.0):
    with io.open(osm, mode="r", encoding="utf-8") as f:  # no Windows o .osm tem que ser lido como UTF-8
        dados = f.read()
    if lat is not None and margem > 0:
        dados = recortar_osm(dados, lat, lon, margem)
    cfg = carla.Osm2OdrSettings()
    cfg.set_osm_way_types(TIPOS_DE_RUA)  # sem calçada, ciclovia, escada
    xodr = carla.Osm2Odr.convert(dados, cfg)
    (AQUI / "mapas" / "karlsruhe_kitti.xodr").write_text(xodr, encoding="utf-8")
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=500.0,
        wall_height=0.0,       # a doc recomenda 0: no OSM, as duas mãos da rua viram ruas separadas e as paredes se cruzam
        additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True)
    return client.generate_opendrive_world(xodr, params), xodr


def tmerc(lat, lon):
    """Transversa de Mercator do PROJ com os padrões ("+proj=tmerc": lon_0=0, k=1, sem falso leste),
    que é o que o Osm2Odr usa. Série de Krüger (Karney 2011), erro de milímetros. Devolve (leste, norte) em m."""
    a, f = 6378137.0, 1 / 298.257223563
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alfa = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240]
    phi, lam = np.radians(lat), np.radians(lon)
    c = 2 * math.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi, eta = np.arctan2(t, np.cos(lam)), np.arctanh(np.sin(lam) / np.sqrt(1 + t ** 2))
    leste = eta + sum(al * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, al in enumerate(alfa, 1))
    norte = xi + sum(al * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, al in enumerate(alfa, 1))
    return A * leste, A * norte


def geo_para_carla(xodr):
    """lat/lon → (x, y) do CARLA. Não dá pra usar cmap.transform_to_geolocation: o CARLA não entende o
    geoReference "+proj=tmerc" que o Osm2Odr escreve e cai num padrão (a rota caía a ~944 km das ruas).
    Então: mesma projeção do conversor + o <offset> do .xodr, e y trocado de sinal (CARLA é mão esquerda)."""
    off = ET.fromstring(xodr.encode("utf-8")).find("header/offset")
    ox, oy = float(off.get("x")), float(off.get("y"))

    def conv(lat, lon):
        e, n = tmerc(lat, lon)
        return e + ox, -(n + oy)
    return conv


def rota(cmap, xodr, arq):
    d = np.loadtxt(arq, delimiter=",")
    t, lat, lon, v, giro = d[:, 0], d[:, 1], d[:, 2], d[:, 4], d[:, 5]
    x, y = geo_para_carla(xodr)(lat, lon)
    # teste: a rota cai em cima de rua?
    dist = []
    for a, b in zip(x[::5], y[::5]):
        w = cmap.get_waypoint(carla.Location(float(a), float(b), 0.0), project_to_road=True)
        dist.append(math.hypot(w.transform.location.x - a, w.transform.location.y - b))
    print(f"rota do KITTI até o centro da faixa mais próxima: mediana {np.median(dist):.1f} m, máximo {np.max(dist):.1f} m")
    return t, x, y, v, giro


def desenhar(world, x, y, segundos=600.0):
    """Rota do KITTI como pontinhos no chão (antes: linha grossa a 0,5 m, que brilhava e escondia o carro)."""
    for i in range(0, len(x), 2):
        world.debug.draw_point(carla.Location(float(x[i]), float(y[i]), 0.1), size=0.06,
                               color=carla.Color(200, 30, 30), life_time=segundos)


def dist_polilinha(px, py, x, y, i, janela=20):
    """Distância do carro até a rota (segmentos entre pontos), não até o ponto mais perto.
    Medir até o ponto dava um serrote falso: com pontos a ~0,8 m, mesmo em cima da linha o erro ia de 0 a ~0,4 m."""
    a, b = max(0, i - janela), min(len(x) - 1, i + janela)
    x0, y0, dx, dy = x[a:b], y[a:b], x[a + 1:b + 1] - x[a:b], y[a + 1:b + 1] - y[a:b]
    u = np.clip(((px - x0) * dx + (py - y0) * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
    return float(np.min(np.hypot(x0 + u * dx - px, y0 + u * dy - py)))


def pure_pursuit(px, py, yaw, x, y, i0, ld, L):
    """Degrau 3: acha a cenoura a ld metros à frente na rota e devolve (δ em rad, índice mais próximo)."""
    janela = slice(i0, min(i0 + 60, len(x)))       # só procura pra frente (a rota não se cruza)
    i = i0 + int(np.argmin(np.hypot(x[janela] - px, y[janela] - py)))
    j = i
    while j < len(x) - 1 and math.hypot(x[j] - px, y[j] - py) < ld:
        j += 1
    # ângulo da cenoura no referencial do carro. CARLA: y pra direita, yaw horário → alpha positivo = virar à direita
    alpha = math.atan2(y[j] - py, x[j] - px) - yaw
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    return math.atan(2 * L * math.sin(alpha) / ld), i


def dirigir(world, x, y, v_kitti, giro_kitti, carro, video, localizar=None):
    """localizar: None (pose perfeita do CARLA) ou dict(vies_graus, escala_v, estimar_vies, semente)."""
    bp = world.get_blueprint_library().find(carro)
    yaw0 = math.degrees(math.atan2(y[5] - y[0], x[5] - x[0]))
    wp = world.get_map().get_waypoint(carla.Location(float(x[0]), float(y[0]), 0.0))
    car = world.spawn_actor(bp, carla.Transform(carla.Location(float(x[0]), float(y[0]), wp.transform.location.z + 0.5),
                                                 carla.Rotation(yaw=yaw0)))  # x, y da rota; só a altura vem da rua
    fis = car.get_physics_control()
    w = fis.wheels
    L = math.hypot((w[0].position.x + w[1].position.x - w[2].position.x - w[3].position.x) / 2,
                   (w[0].position.y + w[1].position.y - w[2].position.y - w[3].position.y) / 2) / 100.0
    max_steer = math.radians(w[0].max_steer_angle)
    cam = escritor = None
    if video:
        import imageio
        cbp = world.get_blueprint_library().find("sensor.camera.rgb")
        cbp.set_attribute("image_size_x", "1280")
        cbp.set_attribute("image_size_y", "720")
        for attr, val in (("bloom_intensity", "0"), ("lens_flare_intensity", "0")):  # a linha vermelha "brilhava" e lavava a imagem
            if cbp.has_attribute(attr):
                cbp.set_attribute(attr, val)
        cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-25)), attach_to=car)
        fila = queue.Queue()
        cam.listen(fila.put)
        escritor = imageio.get_writer("osm_kitti.mp4" if localizar is None else "osm_kitti_ekf.mp4", fps=int(round(1 / DT)))
    world.tick()
    if cam:
        while not fila.empty():
            fila.get()

    log, i, integral = [], 0, 0.0
    esp = world.get_spectator()
    tr0 = car.get_transform()
    yaw_ant = math.radians(tr0.rotation.yaw)
    f = None
    if localizar:
        rng = np.random.default_rng(localizar["semente"])
        vies = math.radians(localizar["vies_graus"])
        f = EKF(tr0.location.x, tr0.location.y, yaw_ant, SIGMA_GPS, SIGMA_V, SIGMA_GIRO,
                estimar_vies=localizar["estimar_vies"])  # largada conhecida, depois é com os sensores
    try:
        for passo in range(int(80 / DT)):  # teto de 80 s (o KITTI levou 46)
            tr = car.get_transform()
            vel = car.get_velocity()
            v = math.hypot(vel.x, vel.y)
            vx, vy, vyaw = tr.location.x, tr.location.y, math.radians(tr.rotation.yaw)  # a VERDADE
            giro = math.atan2(math.sin(vyaw - yaw_ant), math.cos(vyaw - yaw_ant)) / DT  # giro real, na convenção do CARLA
            yaw_ant = vyaw
            if f is None:
                px, py, yaw = vx, vy, vyaw  # cola: pose perfeita do simulador
            else:
                # sensores ruins, simulados a partir do movimento real
                v_roda = v * (1 + localizar["escala_v"]) + rng.normal(0, 0.1)
                giro_med = giro + vies + rng.normal(0, SIGMA_GIRO)
                if passo > 0:
                    f.prever(v_roda, giro_med, DT)
                if passo % GPS_A_CADA == 0 and passo > 0:
                    f.corrigir(vx + rng.normal(0, SIGMA_GPS), vy + rng.normal(0, SIGMA_GPS))
                px, py, yaw = f.s[0], f.s[1], f.s[2]
                v = max(v_roda, 0.0)  # o controle de velocidade também só vê o sensor
            delta, i = pure_pursuit(px, py, yaw, x, y, i, max(LD_MIN, LD_GANHO * v), L)
            if i >= len(x) - 3:
                break
            v_ref = v_kitti[i]          # mesma velocidade que o KITTI andou nesse ponto
            erro = v_ref - v
            integral += erro * DT
            thr = float(np.clip(0.5 * erro + 0.1 * integral, 0.0, 1.0))
            brk = float(np.clip(-0.5 * erro, 0.0, 1.0)) if erro < -1.0 else 0.0
            car.apply_control(carla.VehicleControl(throttle=thr, brake=brk,
                                                   steer=float(np.clip(delta / max_steer, -1, 1))))
            world.tick()
            if cam:
                escritor.append_data(np.frombuffer(fila.get(timeout=5.0).raw_data, np.uint8).reshape(720, 1280, 4)[:, :, [2, 1, 0]])
            esp.set_transform(carla.Transform(tr.location + carla.Location(z=60), carla.Rotation(pitch=-90)))
            lateral = dist_polilinha(vx, vy, x, y, i)  # erro de verdade: posição REAL até a rota
            erro_est = math.hypot(px - vx, py - vy)       # quanto o carro acha que está longe de onde está
            vies_est = math.degrees(f.s[3]) if f is not None and f.n == 4 else 0.0
            d_humano = math.atan(L_KITTI * giro_kitti[i] / max(v_kitti[i], 0.5))  # KITTI: giro anti-horário
            d_roda = math.radians(car.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
            log.append((passo * DT, lateral, v, v_ref, d_roda, d_humano, i, vx, vy, erro_est, vies_est))
    finally:
        if cam:
            cam.stop()
            cam.destroy()
            escritor.close()
            print("vídeo: osm_kitti.mp4" if localizar is None else "vídeo: osm_kitti_ekf.mp4")
        car.destroy()
    return np.array(log), L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--osm", default=str(AQUI / "mapas" / "karlsruhe_kitti.osm"))
    p.add_argument("--rota", default=str(AQUI / "mapas" / "kitti_0009_rota.csv"))
    p.add_argument("--carro", default="vehicle.tesla.model3")
    p.add_argument("--dirigir", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--localizar", action="store_true", help="usa o EKF com sensores ruins em vez da pose perfeita")
    p.add_argument("--vies-giro", type=float, default=0.5, help="viés do giroscópio simulado, °/s (chute: celular)")
    p.add_argument("--escala-v", type=float, default=0.0, help="erro de escala da roda, ex.: 0.02 = 2%%")
    p.add_argument("--sem-vies", action="store_true", help="EKF sem o viés no estado (pra comparar)")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--margem", type=float, default=300.0,
                   help="metros de rua em volta da rota (0 = mapa inteiro, que costuma derrubar o CARLA)")
    a = p.parse_args()

    d = np.loadtxt(a.rota, delimiter=",")
    client = carla.Client(a.host, a.port)
    client.set_timeout(300.0)  # gerar o mapa demora
    world, xodr = carregar_mapa(client, a.osm, d[:, 1], d[:, 2], a.margem)
    client.set_timeout(60.0)
    world.set_weather(carla.WeatherParameters.ClearNoon)  # o mundo gerado vinha com névoa clara
    cmap = world.get_map()
    t, x, y, v, giro = rota(cmap, xodr, a.rota)
    desenhar(world, x, y)
    meio = len(x) // 2
    world.get_spectator().set_transform(carla.Transform(carla.Location(float(x[meio]), float(y[meio]), 250),
                                                        carla.Rotation(pitch=-90)))
    if not a.dirigir:
        print("mapa carregado e rota desenhada em vermelho. Olhe na janela do CARLA.")
        return

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = DT
    world.apply_settings(s)
    try:
        loc = dict(vies_graus=a.vies_giro, escala_v=a.escala_v, estimar_vies=not a.sem_vies,
                   semente=a.semente) if a.localizar else None
        log, L = dirigir(world, x, y, v, giro, a.carro, a.video, loc)
    finally:
        try:
            world.tick()
        except RuntimeError:
            pass
        world.apply_settings(original)

    tt, lat_err, vv, vref, d_roda, d_hum, idx, cx, cy, err_est, vies_est = log.T
    modo = (f"EKF ({'sem' if a.sem_vies else 'com'} viés no estado; giroscópio com {a.vies_giro}°/s, roda {a.escala_v:+.0%})"
            if a.localizar else "pose perfeita do CARLA")
    print(f"modo: {modo}")
    if a.localizar:
        print(f"erro da estimativa de posição: média {err_est.mean():.2f} m, máximo {err_est.max():.2f} m"
              + (f" | viés estimado no fim {vies_est[-1]:.2f}°/s (verdade {a.vies_giro})" if not a.sem_vies else ""))
    print(f"pure pursuit na rota do KITTI: {tt[-1]:.0f} s (KITTI: {t[-1]:.0f} s), "
          f"erro lateral médio {lat_err.mean():.2f} m, máximo {lat_err.max():.2f} m, chegou ao índice {int(idx[-1])}/{len(x)}")
    # CARLA mede o ângulo da roda positivo = direita; KITTI giro positivo = esquerda → troca o sinal pra comparar
    print(f"volante: meu |δ| máximo {np.degrees(np.abs(d_roda)).max():.1f}°, "
          f"humano (deduzido) {np.degrees(np.abs(d_hum)).max():.1f}°")
    sufixo = ("_ekf" + ("_semvies" if a.sem_vies else "") + (f"_escala{a.escala_v:g}" if a.escala_v else "")) if a.localizar else ""
    np.savetxt(f"osm_kitti{sufixo}_log.csv", log, delimiter=",",
               header="t,erro_lateral,v,v_ref,delta_roda_rad,delta_humano_rad,indice,x,y,erro_estimativa,vies_estimado_graus")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(tt, lat_err, label="erro lateral real (onde o carro está × rota)")
        if a.localizar:
            ax[0].plot(tt, err_est, label="erro da estimativa (onde ele acha que está × onde está)", alpha=0.7)
            ax[0].legend(fontsize=8)
        ax[0].set_ylabel("erro (m)")
        ax[0].grid(True)
        ax[1].plot(tt, -np.degrees(d_roda), label="meu pure pursuit (roda medida no CARLA)")
        ax[1].plot(tt, np.degrees(d_hum), label="motorista do KITTI (deduzido do giro)")
        ax[1].set_ylabel("volante (°, + = esquerda)")
        ax[1].set_xlabel("tempo (s)")
        ax[1].legend()
        ax[1].grid(True)
        fig.suptitle(f"Pure pursuit na rota real do KITTI, ruas do OSM: {modo}", fontsize=10)
        fig.savefig(f"osm_kitti{sufixo}.png", dpi=120, bbox_inches="tight")
        print(f"gráfico: osm_kitti{sufixo}.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
