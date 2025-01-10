# mavdsk-python
Código MAVSDK-python para implementar identificação de sistemas em SITL (software in the Loop). O workflow é o seguinte:

1. PX4 rodando em SITL `make px4_sitl gazebo-classic`, testado em:
   * WSL2 em Windows 11,
   * Debian 12 using `px4io/px4-dev-ros-noetic:latest` docker image. Run the `PX4-docker.sh` script,
2. O codigo MAVSDK-Python deste repositório roda em Python `venv`:
   * `python3 -m venv venv`, a primeira vez,
   * `source venv/bin/activate`, para ativar o virtual environment,
   * `pip install -r requirements.txt`, a primeira vez.
   * `deactivate` para encerrar o `venv`.
3. São enviados comandos de atitude utilizando o modo `offboard`, para analizar as dinâmicas das velocidades angulares $p(t)$, $q(t)$ e $r(t)$. 

---

**O que tem sido feito até o momento:**

1. O script `offboard_attitude.py` produz vários sinais de de teste:
    * Sinal dente-de-serra,
    * Sinal quadrado (rico em harmônicos),
    * Sinal triangular,
    * Sinal de varredura de frequência (frequency sweep).
    * Sinal senoidal puro para baixas frequências

2. O sinal é enviado via `MAVlink`, para o PX4 em SITL.

3. Utilizando `QGroundControl` são obtidos os arquivos de dados `ulg`.

4. No Python notebook `XXX_XXX_analise_resposta_em_freq.ipynb` é obtida a resposta em frequência.

5. Os dados gerados de resposta em frequência, assim como os dados temporais, são guardados em arquivos de texto (`txt`).

6. No script `XXX_XXX_analise_resposta_em_freq.ipynb` é obtinda a resposta em frequência a partir dos dados anteriores.

7. É proposto um modelo linear de primeira ordem com atraso 
   
$$\dfrac{P(s)}{U(s)}=ke^{-\tau s}\left(\dfrac{Ts+1}{aTs+1}\right)\left(\frac{1}{I_{xx}s}\right),$$ 

para a dinâmica da velocidade angular $p(t)$.

9. O modelo estimado utilizando Mínimos Quadrados é
    
$$\dfrac{P(s)}{U(s)}=Ke^{-\tau s}\left(\dfrac{s+a}{s+b}\right)\left(\frac{1}{s}\right),$$

11. O modelo foi validado com sucesso.

---

