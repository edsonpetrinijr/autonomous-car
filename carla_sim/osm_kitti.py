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
"""
import argparse
import io
import math
import queue
from pathlib import Path

import carla
import numpy as np

AQUI = Path(__file__).parent
DT = 0.05            # s, modo síncrono (igual ao circulo.py)
L_KITTI = 2.71       # m: entre-eixos do VW Passat B6 do KITTI. Ficha técnica de memória, VERIFICAR antes de citar
LD_MIN, LD_GANHO = 4.0, 0.8  # lookahead = max(4 m, 0,8 s × v). Chute; mesma lógica do degrau 3 (ld cresce com v)
TIPOS_DE_RUA = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential"]


def carregar_mapa(client, osm):
    with io.open(osm, mode="r", encoding="utf-8") as f:  # no Windows o .osm tem que ser lido como UTF-8
        dados = f.read()
    cfg = carla.Osm2OdrSettings()
    cfg.set_osm_way_types(TIPOS_DE_RUA)  # sem calçada, ciclovia, escada
    xodr = carla.Osm2Odr.convert(dados, cfg)
    (AQUI / "mapas" / "karlsruhe_kitti.xodr").write_text(xodr, encoding="utf-8")
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=500.0,
        wall_height=0.0,       # a doc recomenda 0: no OSM, as duas mãos da rua viram ruas separadas e as paredes se cruzam
        additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True)
    return client.generate_opendrive_world(xodr, params)


def geo_para_carla(cmap):
    """Ajusta uma transformação afim lat/lon → (x, y) do CARLA, usando o caminho inverso que a API oferece.
    (transform_to_geolocation existe no 0.9.x; o inverso nem sempre.) Na escala de 2 km, afim basta."""
    wps = cmap.generate_waypoints(20.0)
    xs = np.array([w.transform.location.x for w in wps])
    ys = np.array([w.transform.location.y for w in wps])
    gx, gy = np.meshgrid(np.linspace(xs.min(), xs.max(), 15), np.linspace(ys.min(), ys.max(), 15))
    geo = [cmap.transform_to_geolocation(carla.Location(float(a), float(b), 0.0)) for a, b in zip(gx.ravel(), gy.ravel())]
    A = np.c_[[g.latitude for g in geo], [g.longitude for g in geo], np.ones(len(geo))]
    cx, rx = np.linalg.lstsq(A, gx.ravel(), rcond=None)[:2]
    cy, ry = np.linalg.lstsq(A, gy.ravel(), rcond=None)[:2]
    resid = math.sqrt((rx[0] + ry[0]) / len(geo)) if len(rx) and len(ry) else 0.0
    print(f"ajuste lat/lon → CARLA: resíduo {resid:.3f} m (perto de 0 = projeção quase plana, ok)")
    return lambda lat, lon: (np.c_[lat, lon, np.ones_like(lat)] @ cx, np.c_[lat, lon, np.ones_like(lat)] @ cy)


def rota(cmap, arq):
    d = np.loadtxt(arq, delimiter=",")
    t, lat, lon, v, giro = d[:, 0], d[:, 1], d[:, 2], d[:, 4], d[:, 5]
    x, y = geo_para_carla(cmap)(lat, lon)
    # teste: a rota cai em cima de rua?
    dist = []
    for a, b in zip(x[::5], y[::5]):
        w = cmap.get_waypoint(carla.Location(float(a), float(b), 0.0), project_to_road=True)
        dist.append(math.hypot(w.transform.location.x - a, w.transform.location.y - b))
    print(f"rota do KITTI até o centro da faixa mais próxima: mediana {np.median(dist):.1f} m, máximo {np.max(dist):.1f} m")
    return t, x, y, v, giro


def desenhar(world, x, y, segundos=600.0):
    for i in range(0, len(x) - 3, 3):
        world.debug.draw_line(carla.Location(float(x[i]), float(y[i]), 0.5),
                              carla.Location(float(x[i + 3]), float(y[i + 3]), 0.5),
                              thickness=0.3, color=carla.Color(255, 0, 0), life_time=segundos)


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


def dirigir(world, x, y, v_kitti, giro_kitti, carro, video):
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
        cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=-10, z=7), carla.Rotation(pitch=-25)), attach_to=car)
        fila = queue.Queue()
        cam.listen(fila.put)
        escritor = imageio.get_writer("osm_kitti.mp4", fps=int(round(1 / DT)))
    world.tick()
    if cam:
        while not fila.empty():
            fila.get()

    log, i, integral = [], 0, 0.0
    esp = world.get_spectator()
    try:
        for passo in range(int(80 / DT)):  # teto de 80 s (o KITTI levou 46)
            tr = car.get_transform()
            vel = car.get_velocity()
            v = math.hypot(vel.x, vel.y)
            px, py, yaw = tr.location.x, tr.location.y, math.radians(tr.rotation.yaw)
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
            lateral = float(np.min(np.hypot(x[max(0, i - 20):i + 20] - px, y[max(0, i - 20):i + 20] - py)))
            d_humano = math.atan(L_KITTI * giro_kitti[i] / max(v_kitti[i], 0.5))  # KITTI: giro anti-horário
            d_roda = math.radians(car.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
            log.append((passo * DT, lateral, v, v_ref, d_roda, d_humano, i))
    finally:
        if cam:
            cam.stop()
            cam.destroy()
            escritor.close()
            print("vídeo: osm_kitti.mp4")
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
    a = p.parse_args()

    client = carla.Client(a.host, a.port)
    client.set_timeout(60.0)  # gerar o mapa demora
    world = carregar_mapa(client, a.osm)
    cmap = world.get_map()
    t, x, y, v, giro = rota(cmap, a.rota)
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
        log, L = dirigir(world, x, y, v, giro, a.carro, a.video)
    finally:
        try:
            world.tick()
        except RuntimeError:
            pass
        world.apply_settings(original)

    tt, lat_err, vv, vref, d_roda, d_hum, idx = log.T
    print(f"pure pursuit na rota do KITTI: {tt[-1]:.0f} s (KITTI: {t[-1]:.0f} s), "
          f"erro lateral médio {lat_err.mean():.2f} m, máximo {lat_err.max():.2f} m, chegou ao índice {int(idx[-1])}/{len(x)}")
    # CARLA mede o ângulo da roda positivo = direita; KITTI giro positivo = esquerda → troca o sinal pra comparar
    print(f"volante: meu |δ| máximo {np.degrees(np.abs(d_roda)).max():.1f}°, "
          f"humano (deduzido) {np.degrees(np.abs(d_hum)).max():.1f}°")
    np.savetxt("osm_kitti_log.csv", log, delimiter=",", header="t,erro_lateral,v,v_ref,delta_roda_rad,delta_humano_rad,indice")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(tt, lat_err)
        ax[0].set_ylabel("erro lateral (m)")
        ax[0].grid(True)
        ax[1].plot(tt, -np.degrees(d_roda), label="meu pure pursuit (roda medida no CARLA)")
        ax[1].plot(tt, np.degrees(d_hum), label="motorista do KITTI (deduzido do giro)")
        ax[1].set_ylabel("volante (°, + = esquerda)")
        ax[1].set_xlabel("tempo (s)")
        ax[1].legend()
        ax[1].grid(True)
        fig.suptitle("Pure pursuit dirigindo a rota real do KITTI, nas ruas do OSM")
        fig.savefig("osm_kitti.png", dpi=120, bbox_inches="tight")
        print("gráfico: osm_kitti.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
