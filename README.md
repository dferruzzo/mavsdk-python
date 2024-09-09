# mavdsk-python
Código MAVSDK-python para implementar identificação de sistemas em SITL (software in the Loop). O workflow é o seguinte:

1. PX4 rodando em SITL `make px4_sitl jmavsim` em WSL2 em Windows 11,
2. O codigo MAVSDK-Python deste repositório roda em Ubuntu WSL2,
3. São enviados comandos de atitude utilizando o modo `offboard`, para analizar a taxa de rolagem `p`. 

---

**O que tem sido feito até o momento:**

1. O script `offboard_attitude.py` tem vários sinais de de teste:
    * Sinal dente-de-serra,
    * Sinal quadrado (rico em harmônicos),
    * Sinal triangular,
    * Sinal de varredura de frequência (frequency sweep).

2. O sinal é enviado via `MAVlink`, para o PX4 em SITL.

3. Utilizando `QGroundControl` é obtido o arquivo de dados `ulg`.

4. No Notebook `analise_resposta_em_freq.ipynb` é obtida a resposta em frequência.

5. Os dados gerados de resposta em frequência são guardados em arquivos de texto (`txt`).

6. No script `analise_resposta_em_freq.ipynb` é obtinda a resposta em frequência a partir dos dados anteriores.

7. É proposto um modelo linear de primeira ordem 
$$\dfrac{P(s)}{U(s)}=ke^{-\tau s}\left(\dfrac{Ts+1}{aTs+1}\right)\left(\frac{1}{I_{xx}s}\right),$$ 
para a dinâmica da velocidade angular $p(t)$.

8. O modelo estimado é 
$$\dfrac{P(s)}{U(s)}=Ke^{-\tau s}\left(\dfrac{s+a}{s+b}\right)\left(\frac{1}{s}\right),$$

9. O modelo foi validado com sucesso.

