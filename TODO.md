# TODO

(10/01/2025) Parece existir uma incongruencia entre os dados obtidos a partir do sinal quadrado e os dados obtidos a partir dos sinais senoidais.

- [x] Verificar qual o sinal de control que é para utilizar.
  - timestamps_torque, torque_roll   = get_ulog_data(ulog, 'vehicle_torque_setpoint', 'xyz[0]')?
- [x] Gerar novos dados para o sinal senoidal de frequencia 1.6 (rad/s) com maior amplitude.
- [x] Coletar dados para 10 (rad/s), 100 (rad/s).
- [x] Verificar os dados coletados.