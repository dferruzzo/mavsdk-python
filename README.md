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

---

**To-Do:**

1. Fazer a identificação de sistemas a partir da resposta em frequência.
2. Obter o modelo linear para as taxas;
2. Testar o modelo.

---

