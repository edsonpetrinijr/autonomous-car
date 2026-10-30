"""Degrau 1 no CARLA: volante travado, velocidade constante, medir o raio do círculo.

No nosso modelo de bicicleta (sim/bicicleta.py), o raio NÃO muda com a velocidade: R = L / tan δ.
Aqui a física é de verdade (pneu, peso, suspensão). Pergunta: o raio muda?

Feito pra CARLA 0.9.13+ (usa get_wheel_steer_angle). Testado em 29/out/2026 no lab: CARLA 0.9.16,
Python 3.10 (`py -3.10`; o pacote carla não instala no 3.13), com --mapa Town10HD_Opt --limpar.

Como rodar (na máquina Windows do lab):
  1. Abrir o CARLA (CarlaUE4.exe) e esperar o mapa carregar.
  2. pip install carla==<mesma versão do simulador> numpy matplotlib
     (ou instalar o .whl/.egg que vem em PythonAPI/carla/dist)
  3. python circulo.py                       # 10, 20 e 30 km/h, volante em 40%
     python circulo.py --vel 10 20 30 40 --steer 0.3 --spawn 5
     python circulo.py --mapa Town10HD_Opt --limpar   # tira prédios e postes do caminho
     python circulo.py --mapa Town10HD_Opt --limpar --vel 30 50 --video   # mp4 visto de cima (pip install imageio[ffmpeg])
"""
import argparse
import math
import queue

import carla
import numpy as np

DT = 0.05  # s: passo fixo do modo síncrono (20 Hz). Escolha minha, mesmo papel do DT do nosso sim


def ajustar_circulo(pts):
    """Círculo que melhor passa pelos pontos (mínimos quadrados): x² + y² + D·x + E·y + F = 0."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    D, E, F = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    return math.sqrt(cx**2 + cy**2 - F)


def entre_eixos(car):
    """L em metros, a partir da posição das rodas (o CARLA dá em cm, no mundo)."""
    w = car.get_physics_control().wheels
    fx, fy = (w[0].position.x + w[1].position.x) / 2, (w[0].position.y + w[1].position.y) / 2
    tx, ty = (w[2].position.x + w[3].position.x) / 2, (w[2].position.y + w[3].position.y) / 2
    return math.hypot(fx - tx, fy - ty) / 100.0


def camera_de_cima(world, spawn, altura=50.0):
    """Câmera parada no céu em cima da largada, olhando pra baixo. Vê ~100 m de largura a 50 m de altura."""
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "1280")
    bp.set_attribute("image_size_y", "720")
    bp.set_attribute("fov", "90")
    pose = carla.Transform(spawn.location + carla.Location(z=altura), carla.Rotation(pitch=-90))
    cam = world.spawn_actor(bp, pose)
    fila = queue.Queue()
    cam.listen(fila.put)
    return cam, fila


def para_rgb(img):
    """Imagem do CARLA (BGRA) → array RGB."""
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return a[:, :, [2, 1, 0]]


def rodar(world, bp, spawn, v_kmh, steer, tempo, video=False):
    v_ref = v_kmh / 3.6
    car = world.spawn_actor(bp, spawn)
    col = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                            carla.Transform(), attach_to=car)
    bateu = []
    col.listen(lambda e: bateu.append(e.other_actor.type_id))
    cam = escritor = None
    if video:
        import imageio  # pip install imageio[ffmpeg]
        cam, fila = camera_de_cima(world, spawn)
        nome = f"circulo_{v_kmh:.0f}kmh.mp4"
        escritor = imageio.get_writer(nome, fps=int(round(1 / DT)))  # 20 fps = tempo real
    world.tick()
    if cam:
        while not fila.empty():  # descarta o quadro do tick de largada
            fila.get()
    L = entre_eixos(car)
    espectador = world.get_spectator()

    pts, angs, alats, vs = [], [], [], []
    integral = 0.0
    for i in range(int(tempo / DT)):
        vel = car.get_velocity()
        v = math.hypot(vel.x, vel.y)
        # controle de velocidade PI: acelerador proporcional ao erro + acumulado (senão sobra erro de regime, degrau 6)
        erro = v_ref - v
        integral += erro * DT
        throttle = float(np.clip(0.5 * erro + 0.1 * integral, 0.0, 1.0))
        brake = float(np.clip(-0.5 * erro, 0.0, 1.0)) if erro < -1.0 else 0.0
        car.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        world.tick()
        if cam:
            escritor.append_data(para_rgb(fila.get(timeout=5.0)))

        loc = car.get_location()
        espectador.set_transform(carla.Transform(loc + carla.Location(z=45), carla.Rotation(pitch=-90)))
        if i * DT > tempo / 2:  # mede só a 2ª metade (já em velocidade)
            pts.append((loc.x, loc.y))
            fl = car.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
            fr = car.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
            angs.append(abs(fl + fr) / 2)
            alats.append(v**2)  # dividido pelo raio depois
            vs.append(v)
        if bateu:
            break

    if cam:
        cam.stop()
        cam.destroy()
        escritor.close()
        print(f"vídeo: {nome}")
    col.destroy()
    car.destroy()
    world.tick()
    if len(pts) < 20:
        return dict(v_kmh=v_kmh, bateu=bateu[0] if bateu else "?", L=L)
    pts = np.array(pts)
    delta = math.radians(float(np.mean(angs)))  # ângulo REAL da roda, medido
    r_med = ajustar_circulo(pts)
    return dict(v_kmh=v_kmh, L=L, delta_graus=math.degrees(delta), r_teo=L / math.tan(delta),
                r_med=r_med, a_lat=float(np.mean(alats)) / r_med, bateu=bateu[0] if bateu else None, pts=pts,
                v_med=float(np.mean(vs)) * 3.6)  # se ficar bem abaixo do pedido, o pneu saturou


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--mapa", default=None, help="ex.: Town10HD_Opt (padrão: o mapa já aberto)")
    p.add_argument("--limpar", action="store_true", help="tira prédios/postes/árvores (só mapas _Opt)")
    p.add_argument("--carro", default="vehicle.tesla.model3")
    p.add_argument("--steer", type=float, default=0.4, help="volante de −1 a 1. 0,4: chute")
    p.add_argument("--vel", type=float, nargs="+", default=[10, 20, 30], help="km/h")
    p.add_argument("--spawn", type=int, default=0, help="índice do ponto de largada do mapa")
    p.add_argument("--tempo", type=float, default=20.0, help="s por velocidade")
    p.add_argument("--video", action="store_true", help="grava um mp4 por velocidade, câmera de cima")
    a = p.parse_args()

    client = carla.Client(a.host, a.port)
    client.set_timeout(20.0)
    world = client.get_world()
    if a.mapa and not world.get_map().name.endswith(a.mapa):  # só recarrega se for outro mapa
        world = client.load_world(a.mapa)
    if a.limpar:
        # uma camada por vez: no 0.9.16, MapLayer.A | MapLayer.B vira int e a chamada falha
        for camada in (carla.MapLayer.Buildings, carla.MapLayer.Props, carla.MapLayer.StreetLights,
                       carla.MapLayer.Foliage, carla.MapLayer.ParkedVehicles, carla.MapLayer.Walls):
            world.unload_map_layer(camada)

    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode = True   # o simulador só anda quando eu mando (world.tick): resultado repetível
    s.fixed_delta_seconds = DT
    world.apply_settings(s)

    bp = world.get_blueprint_library().find(a.carro)
    spawn = world.get_map().get_spawn_points()[a.spawn]
    resultados = []
    try:
        for v in a.vel:
            r = rodar(world, bp, spawn, v, a.steer, a.tempo, a.video)
            resultados.append(r)
            if "r_med" in r:
                print(f"{v:4.0f} km/h | roda {r['delta_graus']:4.1f}° | L {r['L']:.2f} m | "
                      f"R teórico {r['r_teo']:5.1f} m | R medido {r['r_med']:5.1f} m | "
                      f"a_lat {r['a_lat']:4.1f} m/s² | v medida {r['v_med']:4.1f} km/h"
                      + (f" | BATEU em {r['bateu']}" if r["bateu"] else ""))
            else:
                print(f"{v:4.0f} km/h | BATEU em {r['bateu']} antes de medir. Tente --spawn outro ou --limpar")
    finally:
        # o simulador travava depois do script: um tick extra antes de devolver o modo assíncrono
        try:
            world.tick()
        except RuntimeError:
            pass
        world.apply_settings(original)

    try:
        import matplotlib.pyplot as plt
        for r in resultados:
            if "pts" in r:
                pts = r["pts"] - r["pts"].mean(axis=0)
                plt.plot(pts[:, 0], pts[:, 1], label=f"{r['v_kmh']:.0f} km/h: R = {r['r_med']:.1f} m")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.title(f"CARLA: volante travado em {a.steer}, {a.carro}")
        plt.savefig("circulo_carla.png", dpi=120)
        print("gráfico: circulo_carla.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
