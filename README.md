# mavdsk-python
Código MAVSDK-python para implementar identificação de sistemas em SITL (software in the Loop). O workflow é o seguinte:

1. PX4 rodando em SITL `make px4_sitl none_iris` em WSL2 em Windows 11.
2. PX4 listener rodando em MATLAB 2023 em Windows 11 seguindo as instruções de [https://github.com/dferruzzo/pixhawk-sil-connector.git](https://github.com/dferruzzo/pixhawk-sil-connector.git). O listener provee o modelo do quadrirotor.
3. O codigo MAVSDK-Python deste repositório roda em ubuntu WSL 2.

---

**O que tem sido feito até o momento:**

1. Gerar uma varredura de frequência (freqauency sweep) controlando diretamente a atitude do veículo utlizando o modo Offboard do `MAVSDK`. Script `offboard_attitude.py`. No momento apenas tem sido testado o angulo Roll e a taxa p.

2. Os dados produzidos e capturados no MATLAB são procesados para obter a resposta em frequencia.

3. Para testar o script antes da coleta de dados utilizo `make px4_sitl jmavsim`. 

---

**Desafios:**

1. Gerar a varredura de frequencia utilizando o `ActuatorControl` do MAVSDK que age diretamente nos atuadores. Até o momento, sem sucesso. Script `offboard_actuator_control.py`.
2. A uso direto dos atuadores é necessário para implementar um controlador (linear ou não-linear) de atitude do veículo.