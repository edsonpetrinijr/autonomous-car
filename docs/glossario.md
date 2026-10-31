# Glossário: carro autônomo

Termos na ordem em que apareceram.

## Degrau 1: o carro sozinho (29/out/2026)
- **x, y**: onde o carro **está** agora (medido no eixo traseiro). Não é onde ele quer ir.
- **θ (theta, yaw, heading)**: pra onde o carro aponta. 0 = olhando pro +x.
- **δ (delta, ângulo de esterço)**: quanto a roda da frente está virada em relação ao corpo do carro. 0 = reto; positivo = esquerda. É o volante.
- **L (entre-eixos, wheelbase)**: distância entre o eixo traseiro e o dianteiro. F1TENTH: 0,33 m.
- **ICR (centro instantâneo de rotação)**: o ponto em volta do qual o carro está girando agora. Fica onde as linhas perpendiculares às duas rodas se cruzam.
- **R = L / tan δ**: raio da curva com o volante parado em δ.
- **dθ/dt = v / L · tan δ**: velocidade de giro = velocidade / raio (ω = v / R).
- **Não-holonômico**: não consegue andar em qualquer direção na hora; o carro não anda de lado nem gira parado. O Go2 consegue.
- **Cinemática direta × inversa**: direta = comandos → movimento (o modelo de bicicleta). Inversa = "quero chegar ali" → que comando dar (o que o controlador faz, ex.: pure pursuit).
- **Euler**: integrar somando derivada × dt a cada passo. Simples; erra um pouco, e o erro diminui com dt menor.

## Degrau 2: seguir uma reta (29/out/2026)
- **Controlador**: regra que olha o estado e escolhe o comando (aqui, o volante δ) a cada passo. Fecha o loop.
- **Erro lateral (y)**: distância do carro até a linha que ele quer seguir.
- **Erro de direção (θ)**: ângulo entre pra onde o carro aponta e a direção da linha.
- **Controle P (proporcional)**: δ = −k·erro. Só olha onde está → passa do ponto e oscila pra sempre (pêndulo sem atrito). k maior = oscila mais rápido, não menos.
- **Controle PD**: soma um termo que olha a tendência (aqui, θ: pra onde está indo). É o atrito/amortecedor: desvira antes de chegar.
- **Ganho (k)**: quanto o controlador reage por unidade de erro.
- **Sub-amortecido / crítico / super-amortecido**: amortecimento pouco (balança e morre), no ponto (chega rápido sem passar), demais (chega devagar, rastejando). Aqui, com k_y = 1: crítico em k_θ ≈ 1,15.
- **Linearizar**: trocar a equação de verdade por uma aproximada que vale perto do ponto de interesse (ângulo pequeno: sen θ ≈ θ, tan δ ≈ δ). Vira uma equação que tem fórmula pronta.
- **Massa-mola-amortecedor**: y'' + 2ζω·y' + ω²·y = 0. ω = quão rápido balança; ζ = amortecimento (0 = pra sempre, 1 = crítico, > 1 = lento).
  - Carro seguindo reta com PD: ω = v·√(k_y/L), ζ = k_θ / (2·√(k_y·L)). Crítico (ζ = 1): **k_θ = 2·√(k_y·L)** = 2·√0,33 ≈ 1,15. Não depende de v.
  - Fluxo real: fórmula no modelo linear dá o ponto de partida → testar e ajustar no sistema de verdade (saturação, atraso, pneu escorregando).

## Degrau 3: pure pursuit (29/out/2026)
- **Pure pursuit**: mira num ponto do caminho a ld à frente (cenoura) e vira no arco exato que chega nele: κ = 2·sen α / ld, δ = atan(L·κ). Funciona em qualquer caminho. Único ajuste: ld.
- **Lookahead (ld)**: a distância da cenoura. Grande = suave, mas **corta curva por dentro**. Pequeno = fiel, mas **sensível a atraso**: com atraso, zigue-zagueia.
- **Atraso (latência)**: o comando chega depois (servo, sensor, rede). Quase todo controlador que é ótimo no sim perfeito piora com atraso. Resultado (oval R = 2 m, 1 m/s): sem atraso, melhor ld = 0,3 m (erro máx 2,5 cm); com 0,2 s, melhor ld = 1 m (3,8 cm) e o 0,3 vai a 15,6 cm.
- **Sim-to-real gap**: diferença entre o simulador e o mundo. Cada "trapaça" do sim é uma fonte dela.

## Degrau 4: velocidade e aderência (29/out/2026)
- **Aceleração lateral**: v²/R. É o "empurrão pro lado" que a curva exige; dobrar v = 4× mais.
- **Limite de aderência**: o pneu aguenta até μ·g (F1TENTH: ≈ 10,3 m/s²). Passou disso, escorrega.
- **Understeer (sair de frente)**: escorregou → faz a curva mais aberta e sai **por fora**.
- **Velocidade máxima na curva**: √(μ·g·R). Oval R = 2 m: ≈ 4,5 m/s.
- **Perfil de velocidade**: rápido na reta, no limite na curva. Oval: 4,26 s × 5,46 s da melhor constante (22% mais rápido). Ainda com trapaça: freia na hora (freio infinito).
- **Próximos controladores** (ver abaixo)

## Degrau 5: ruído e filtro (29/out/2026)
- **Estado real × estimado**: onde o carro está × onde o controlador acha que está. Com sensor, nunca são iguais.
- **Ruído de medida**: erro aleatório do sensor a cada leitura (aqui σ = 15 cm).
- **Medida crua**: o carro quase não se perde (2,7 cm × 2,2 cm sem ruído), porque o próprio carro "filtra" (não consegue pular de lado), mas o **volante treme** 40× mais (6,3°/passo). Servo de verdade não aguenta.
- **Média móvel**: tira o ruído, mas a estimativa fica **atrasada** (mostra onde o carro estava) → erro 4× maior (9,4 cm). É o atraso do degrau 3 de novo.
- **Filtro prever-corrigir**: prevê com o modelo (sei v e δ) e corrige um pouco (K) com cada medida. Volante liso e erro quase igual ao sem ruído (2,5 cm). É o esqueleto do **filtro de Kalman**, que escolhe o K sozinho a partir de quanto confia no modelo × no sensor.
- **Odometria / dead reckoning**: só prever, sem corrigir (K = 0). Erro cresce sem parar (drift).

## Degrau 6: LiDAR (29/out/2026)
- **LiDAR 2D**: laser que gira e mede a distância em vários ângulos (F1TENTH: Hokuyo UST-10LX, 270°, 10 m, 40 Hz). O resultado de uma volta do laser é um **scan**: só pontos, sem saber o que é parede, carro ou pessoa.
- **Ray casting**: simular o LiDAR = lançar cada raio e achar a primeira parede que ele cruza.
- **Controle reativo**: decide só pelo que vê agora, sem mapa (aqui: centralizar entre as paredes). Simples e robusto, mas míope.
- **Erro de regime (steady-state error)**: erro que sobra parado. O PD só vira o volante se tiver erro; na curva precisa de volante → precisa de erro → anda ~13 cm pra fora da linha central. Conserto: termo **I** (integral: acumula o erro e empurra até zerar) ou **feedforward** (já virar o volante pela curvatura que vê à frente).
- **Mapa × reativo**: com mapa (pure pursuit, degrau 3) o carro sabe a curva antes de chegar; reativo só sente quando já está nela.

## Meu modelo: imitação (29/out/2026)
- **Behavior cloning (imitação)**: gravar pares (o que o sensor viu, o que o especialista fez) e treinar uma rede pra imitar. Sem regra escrita.
- **Especialista**: quem gera os exemplos (aqui, o pure pursuit com mapa e posição exata).
- **Validação**: separar dados que a rede não vê no treino pra checar se aprendeu ou decorou (aqui: 0,15° de erro no volante).
- **Generalização**: funcionar em situação nova. Resultado: no oval de treino, rede = especialista (2,2 cm). Na pista nova, não bateu, mas errou 21,7 cm × 7,9 cm do especialista, e **94–100% dos erros grandes foram nas curvas pra direita**, que nunca apareceram no treino.
- **Mudança de distribuição (distribution shift)**: a rede vê no teste algo diferente do treino. Consertos: mais variedade de dados (várias pistas, os dois sentidos), **data augmentation** (ex.: espelhar o scan e trocar o sinal do volante = curva pra direita de graça), **DAgger**.
- **Resultado do espelho**: erro máx nas curvas pra direita caiu de 21,7 → 7,8 cm (especialista: 7,9). Mesmos dados, nenhuma coleta nova. Lição: muitas vezes o problema é o que tem nos dados, não a rede.

## Meu modelo: evolução (29/out/2026)
- **MLP (Multilayer Perceptron)**: a rede neural mais simples. Camadas em fila (entrada → escondida → saída); cada neurônio soma as entradas × **pesos** e passa por uma curva (tanh, ReLU). Os pesos são o que se aprende.
- **Neuroevolução**: aprender os pesos por seleção, sem gradiente e sem especialista. População aleatória → nota (fitness) → os melhores viram pais → filhos = pais + mutação aleatória → repete. É o Deep Cars (Universo Programado).
- **Fitness**: a nota que decide quem sobrevive. Aqui: metros andados em 15 s antes de bater. **Você recebe o que pede**: a nota não falava em ficar no centro, então o campeão corta curva (erro lateral ~40 cm).
- **Elitismo**: os melhores passam pra próxima geração sem mudar, então o recorde nunca piora.
- **Resultado** (200 carros, 30 gerações, ~2,5 min no Mac): geração 1 → 199 de 200 bateram. Geração 5 já empatou com o especialista (4,25 s/volta). Geração 30: **3,53 s/volta, 17% mais rápido que o especialista (4,26)**, porque descobriu sozinho a **linha de corrida** (abrir, cortar por dentro, abrir de novo) em vez de seguir o centro. E deu 3,9 voltas no feijão sem bater, sem nunca ter visto.
- **Linha de corrida**: a trajetória mais rápida usa a largura toda da pista; curva de raio maior = mais velocidade (v = √(μ·g·R)).
- **Imitação × evolução**: imitação aprende rápido, mas só fica tão boa quanto o especialista. A evolução não precisa de ninguém e pode superar o especialista, mas precisa de muitas tentativas e só otimiza o que a nota mede.
- **Reward shaping (moldar a nota)**: mudar a nota pra pedir o comportamento que você quer de verdade. Ex.: `metros − 5 × segundos raspando` trocou 4% de velocidade (3,53 → 3,67 s/volta) por folga até a parede (1 → 10 cm). O peso importou menos do que eu previa: 5 e 20 deram o mesmo resultado, e punir ou não já pesou mais que o valor.
- **Semente (seed)**: o sorteio inicial de uma execução. Evolução varia muito entre sementes, então uma execução só não prova nada. Rodar 3 ou mais e olhar a média e a variação é o jeito de separar efeito de ruído.

## Controladores além do pure pursuit
- **Stanley** (erro lateral + direção na roda da frente, o PD refinado; carro de Stanford no DARPA 2005) · **pure pursuit adaptativo/regulado** (ld cresce com v; o do Nav2) · **MPC** (simula o futuro com o modelo e escolhe os comandos respeitando os limites; é o que se usa em corrida).

## Dado real e localização (30/out)
- **KITTI**: dataset de um carro de verdade em Karlsruhe (2011): LiDAR Velodyne, câmeras, IMU e GPS/INS RTK (OXTS), que serve de "verdade" com ~0,3–0,5 m de precisão. Mesmo ele erra: no drive 0005, velocidade e posição discordam ~2,5%.
- **Dead reckoning ("andar às cegas")**: integrar velocidade e giro a partir de uma pose conhecida, sem olhar nada de fora. O erro só cresce: 0,8% da distância com IMU de laboratório, ~11% com sensor comum.
- **RMSE**: raiz do erro quadrático médio. Um número pro erro ao longo do trajeto todo; pune mais os erros grandes. Teste de sanidade: GPS com σ 3 m por eixo dá RMSE 3·√2 ≈ 4,24 m.
- **EKF (filtro de Kalman estendido)**: o prever-corrigir do degrau 5 com incerteza. Prever com o movimento (aumenta a incerteza P), corrigir com a medida (ganho K = P/(P+R): confia em quem tem menos incerteza). "Estendido" = o modelo é curvo (cos, sin) e o filtro usa a derivada (F) em volta do ponto atual.
- **Viés do giroscópio**: erro fixo que o giroscópio soma sempre pro mesmo lado. Ruído some na média; viés não. Solução: pôr o viés como variável do estado e deixar o filtro estimar.
- **"O filtro só corrige o que está no modelo"**: com viés no estado, o erro do viés some (3,31 → 1,83 m); os 2% de escala da roda, que não estavam no modelo, continuaram custando ~3 m.
- **As 4 camadas do carro autônomo**: localização (onde estou?), percepção (o que tem em volta?), planejamento (pra onde vou?), controle (como viro e acelero?). O pure pursuit é controle.
- **OpenStreetMap → OpenDRIVE**: o mapa aberto de ruas vira o formato de estrada do CARLA (`carla.Osm2Odr`). Só ruas, sem prédios.
- **GPS não segura faixa**: GPS de 3 m deu 1,34 m de erro lateral no CARLA e meia faixa ultrapassada 43% do tempo. Carro de verdade se localiza pela faixa (câmera) ou pelo mapa (LiDAR), com centímetros.
