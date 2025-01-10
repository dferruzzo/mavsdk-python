import matplotlib.pyplot as plt
import numpy as np
from myfunctions import (read_ulog, get_ulog_data, butter_lowpass_filter, offboard_time_analysis)

class Parametros():
    """
    Handles the initialization and storage of parameters for signal processing.

    This class is designed to store and manage several parameters required for signal
    processing operations, such as file name, frequency values, and other configurations.
    It includes default values for some attributes and allows modifications through
    instantiation.

    Attributes:
        file_name (str): The name of the file to be processed.
        w0 (float): The angular frequency value.
        f0 (float): The equivalent frequency, calculated as `w0 / (2 * π)`.
        cutoff_freq_1 (float): The first cutoff frequency for the signal control filter.
        cutoff_freq_2 (float): The second cutoff frequency for the signal rates filter.
        filt_ordem (int): The order of the filter, set by default to 5.
        time_wait (float): The delay time used in certain operations.
        plot_fig (bool): A flag that specifies whether to generate plots or not.
    """
    def __init__(self, file_name, w0, cutoff_freq_1, cutoff_freq_2, time_wait, plot_fig):
        self.file_name = file_name
        self.w0 = w0
        self.f0 = w0/(2*np.pi)
        self.cutoff_freq_1 = cutoff_freq_1 # self.f0*20 # FPB para o sinal de controle
        self.cutoff_freq_2 = cutoff_freq_2 # self.f0*15 # FPB para o sinal das taxas
        self.filt_ordem = 5
        self.time_wait = time_wait
        self.plot_fig = plot_fig    # Boolean

def recorta_dados(t, x, t0, t1):
    """
    Recorta um vetor de tempo e um vetor de valores associados baseado em um intervalo especificado.

    This function clips the provided time vector `t` and the associated values vector `x`
    to a specific time range defined by `t0` and `t1`. It returns the clipped vectors
    containing only the values and times within the given range.

    Args:
        t: Array-like time vector from which the interval will be clipped.
        x: Array-like vector of values associated with the time vector `t`.
        t0: Start time of the interval to clip. Values prior to `t0` will be excluded.
        t1: End time of the interval to clip. Values after `t1` will be excluded.

    Returns:
        Tuple containing two elements:
            - t_clipped: Clipped time vector including only values between `t0` and `t1`.
            - x_clipped: Clipped vector of values associated with the clipped time vector.
    """
    t_clipped = t[(t >= t0) & (t <= t1)] # Recortando o vetor de tempo
    x_clipped = x[(t >= t0) & (t <= t1)]
    return t_clipped, x_clipped

def pontos_criticos(t, x):
    """
    Identifies the critical points (local maxima and minima) in a given dataset by analyzing the
    changes in the sign of the second derivative. The function separates critical points into
    local maxima and minima based on the concavity indicated by the second derivative.

    Args:
        t (np.ndarray): 1D NumPy array representing the time or independent variable values.
        x (np.ndarray): 1D NumPy array representing the dependent variable values, corresponding
            to the `t` array.

    Returns:
        tuple: A tuple of NumPy arrays containing the critical points:
            - t_pontos_criticos_min (np.ndarray): Array of time values corresponding to the local minima.
            - x_pontos_criticos_min (np.ndarray): Array of dependent variable values corresponding
                to the local minima.
            - t_pontos_criticos_max (np.ndarray): Array of time values corresponding to the local maxima.
            - x_pontos_criticos_max (np.ndarray): Array of dependent variable values corresponding
                to the local maxima.
    """
    # Identificando as mudanças no sinal da derivada de x
    diff_sign_change = np.diff(np.sign(np.diff(x)))
    idx_pontos_criticos = np.where(diff_sign_change != 0)[0] + 1  # Índices dos pontos críticos
    t_pontos_criticos = t[idx_pontos_criticos]
    x_pontos_criticos = x[idx_pontos_criticos]

    # Segunda derivada para identificar máximos e mínimos
    second_diff = np.diff(np.diff(x))
    second_diff_sign = np.sign(second_diff)

    # Listas vazias para armazenar máximos e mínimos
    t_pontos_criticos_min = []
    x_pontos_criticos_min = []
    t_pontos_criticos_max = []
    x_pontos_criticos_max = []

    # Classificando os pontos críticos em máximos ou mínimos
    for i, idx in enumerate(idx_pontos_criticos):
        if i < len(second_diff_sign):  # Garantir que não acesse índices fora da faixa
            if second_diff_sign[idx] > 0:  # Concavidade positiva -> Mínimo local
                t_pontos_criticos_min.append(t_pontos_criticos[i])
                x_pontos_criticos_min.append(x_pontos_criticos[i])
            elif second_diff_sign[idx] < 0:  # Concavidade negativa -> Máximo local
                t_pontos_criticos_max.append(t_pontos_criticos[i])
                x_pontos_criticos_max.append(x_pontos_criticos[i])

    # Convertendo listas para arrays NumPy
    t_pontos_criticos_min = np.array(t_pontos_criticos_min)
    x_pontos_criticos_min = np.array(x_pontos_criticos_min)
    t_pontos_criticos_max = np.array(t_pontos_criticos_max)
    x_pontos_criticos_max = np.array(x_pontos_criticos_max)

    return t_pontos_criticos_min, x_pontos_criticos_min, t_pontos_criticos_max, x_pontos_criticos_max

def amplitude(t, x):
    """
    Detecta a amplitude de um sinal senoidal (sem componente contínua) puro e retorna seu valor médio.

    Args:
        t (np.ndarray): Vetor de tempo.
        x (np.ndarray): Vetor de valores do sinal.

    Returns:
        float: Amplitude média do sinal.
    """
    # Identifica os pontos críticos
    t_min, pc_min, t_max, pc_max = pontos_criticos(t, x)
    # Calcula a amplitude
    ampl = []
    for i in range(min(len(pc_min), len(pc_max))):
        ampl.append(abs(pc_max[i] - pc_min[i]) / 2)

    return np.mean(ampl) if ampl else 0.0

def ganho(t1, x1, t2, x2):
    """
    Calculates the gain of a signal based on the amplitudes of two signals and returns both the linear gain and the gain in decibels (dB).

    This function computes the amplitude of two given signals, then calculates the ratio of the second signal's amplitude to the first signal's amplitude.
    The result is provided as both a linear ratio and in decibels (dB).

    Args:
        t1: The time domain signal corresponding to x1.
        x1: The first signal for which the amplitude needs to be calculated.
        t2: The time domain signal corresponding to x2.
        x2: The second signal for which the amplitude needs to be calculated.

    Returns:
        Tuple[float, float]: A tuple where the first element is the linear gain between the amplitudes of x2 and x1, and the second element is the gain in decibels (dB).
    """
    amplitude_x1 = amplitude(t1, x1)
    amplitude_x2 = amplitude(t2, x2)
    return amplitude_x2/amplitude_x1, 20*np.log10(amplitude_x2/amplitude_x1)

def cruzamento_zero(t, x):
    """
    Detecta os cruzamentos por zero de um sinal senoidal puro.

    Args:
        t (np.ndarray): Vetor de tempo associado ao sinal.
        x (np.ndarray): Vetor de valores do sinal.

    Returns:
        tuple:
            - t_cross (list): Lista com os instantes de tempo em que os cruzamentos ocorrem.
            - direction (list): Lista com as direções dos cruzamentos:
                +1 para cruzamentos ascendentes (de negativo para positivo),
                -1 para cruzamentos descendentes (de positivo para negativo).
    """
    # Identificando os índices em que ocorrem mudanças de sinal
    sign_changes = np.diff(np.sign(x))  # Diferença nos sinais consecutivos
    idx_cross = np.where(sign_changes != 0)[0]  # Índices dos cruzamentos

    # Lista para armazenar os instantes de tempo e direções dos cruzamentos
    t_cross = []
    direction = []

    for idx in idx_cross:
        # Aproximação linear para encontrar o tempo exato do cruzamento
        t0, t1 = t[idx], t[idx + 1]
        x0, x1 = x[idx], x[idx + 1]
        t_cross_val = t0 - x0 * (t1 - t0) / (x1 - x0)  # Interpolação linear para o tempo
        t_cross.append(t_cross_val)

        # Determinando a direção do cruzamento
        if x1 > x0:
            direction.append(+1)  # Ascendente
        else:
            direction.append(-1)  # Descendente

    return t_cross, direction

def desfasagem(t1, x1, t2, x2):
    """
    Calcula a defasagem, em graus, entre dois sinais senoidais puros.
    x2 respeito de x1.

    Args:
        t1, x1: Vetores de tempo e valores do primeiro sinal.
        t2, x2: Vetores de tempo e valores do segundo sinal.

    Returns:
        float: Defasagem em graus entre os dois sinais.
    """
    # Detectando cruzamentos por zero de ambos os sinais
    cruzamento_zero_1, direction_1 = cruzamento_zero(t1, x1)
    cruzamento_zero_2, direction_2 = cruzamento_zero(t2, x2)

    # Filtrando apenas cruzamentos ascendentes (direction == +1)
    cz1_acima = [cruzamento_zero_1[i] for i in range(len(cruzamento_zero_1)) if direction_1[i] == 1]
    cz2_acima = [cruzamento_zero_2[i] for i in range(len(cruzamento_zero_2)) if direction_2[i] == 1]

    # Garantir que há cruzamentos em ambos os sinais
    if not cz1_acima or not cz2_acima:
        raise ValueError("Não foi possível calcular a defasagem: cruzamento por zero ausente em um dos sinais.")

    # Calculando a diferença de tempo entre o primeiro cruzamento de ambos os sinais
    delta_t = cz2_acima[0] - cz1_acima[0]

    # Calculando o período do sinal (assumindo frequência constante do 1º sinal)
    periodo = np.mean(np.diff(cz1_acima))  # Período total com base no intervalo de tempo fornecido
    #print(f"Período do sinal 1: {periodo:.2f} s")
    #print(f"frequencia do sinal 1: {1/periodo:.2f} Hz")
    #print(f"freqencia do sinal 1 em rad/s: {(1/periodo)*2*np.pi:.2f} rad/s")
    # Convertendo a defasagem de tempo para graus
    desfase_graus = (delta_t / periodo) * 360  # 360 graus por período
    #print(f"Desfase em graus: {desfase_graus:.2f} graus")

    return desfase_graus

def coleta_dados_p_rate(ulog: object) -> object:
    """
    Extracts data related to actuator control signals and angular velocity rates
    from the provided ulog object and computes a control signal value.

    Args:
        ulog (object): The ulog object containing flight data logs.

    Returns:
        tuple: A tuple consisting of:
            - t_controle (list): Timestamps corresponding to control signal data.
            - controle (list): Computed control signal values derived from
              actuator motor data.
            - t_p_rate (list): Timestamps corresponding to angular velocity
              data.
            - p_rate (list): Angular velocity data for the roll axis
              (x-axis component).
    """
    # Dados dos atuadores
    t_controle, control0 = get_ulog_data(ulog, 'actuator_motors', 'control[0]')
    _, control1 = get_ulog_data(ulog, 'actuator_motors', 'control[1]')
    _, control2 = get_ulog_data(ulog, 'actuator_motors', 'control[2]')
    _, control3 = get_ulog_data(ulog, 'actuator_motors', 'control[3]')
    # o sinal de controle
    controle = control1 + control2 - control0 - control3
    # dados das taxas de rotação
    t_p_rate, p_rate = get_ulog_data(ulog, 'vehicle_angular_velocity', 'xyz[0]')
    return t_controle, controle, t_p_rate, p_rate

def filtra_recorta_dados(t_controle, controle, t_rate, rate, cutoff_freq_1, cutoff_freq_2, filt_order,
                         t0_clip, t1_clip):
    """
    Filters and clips signal data based on given parameters.

    This function applies low-pass Butterworth filters to two datasets and subsequently
    clips the filtered data over specified time intervals. The function operates on
    two signals, identified as `controle` and `rate`, along with their corresponding
    time arrays.

    Args:
        t_controle: Time array corresponding to the `controle` signal.
        controle: The signal data to be filtered and clipped.
        t_rate: Time array corresponding to the `rate` signal.
        rate: The signal data to be filtered and clipped.
        cutoff_freq_1: High cutoff frequency for the low-pass filter applied to the
            `controle` signal.
        cutoff_freq_2: High cutoff frequency for the low-pass filter applied to the
            `rate` signal.
        filt_order: Integer denoting the order of the Butterworth low-pass filter.
        t0_clip: Start time for clipping the filtered `controle` and `rate` signals.
        t1_clip: End time for clipping the filtered `controle` and `rate` signals.

    Returns:
        Tuple containing four elements:
            - Clipped time array for the `controle` signal.
            - Clipped data for the `controle` signal.
            - Clipped time array for the `rate` signal.
            - Clipped data for the `rate` signal.

    """
    # filtrar sinais
    dt_controle = t_controle[1] - t_controle[0]
    dt_rate = t_rate[1] - t_rate[0]
    controle_filt = butter_lowpass_filter(controle, cutoff_freq_1, 1 / dt_controle, order=filt_order)
    rate_filt = butter_lowpass_filter(rate, cutoff_freq_2, 1 / dt_rate, order=filt_order)
    # clipando os dados
    t_cont_clipped, controle_clipped = recorta_dados(t_controle, controle_filt, t0_clip, t1_clip)
    t_rate_clipped, rate_clipped = recorta_dados(t_rate, rate_filt, t0_clip, t1_clip)
    return t_cont_clipped, controle_clipped, t_rate_clipped, rate_clipped

def bode_point(t_cont_clipped, controle_clipped, t_rate_clipped, rate_clipped):
    """
    Calculates the Bode plot point based on gain (in dB) and phase shift for given inputs.

    This function computes the gain (in decibels) and phase shift (in degrees) for a control
    transfer function and rate transfer function, provided in a clipped form. The results
    are used to form a point in the Bode plot analysis, which is essential in control
    systems for the study of system stability and response.

    Args:
        t_cont_clipped: Time domain array for the clipped control transfer function.
        controle_clipped: Clipped control transfer function values.
        t_rate_clipped: Time domain array for the clipped rate transfer function.
        rate_clipped: Clipped rate transfer function values.

    Returns:
        tuple: A tuple where the first element is the gain in decibels (float) and the
        second element is the phase shift in degrees (float).

    """
    _, ganho_dB = ganho(t_cont_clipped, controle_clipped, t_rate_clipped, rate_clipped)
    fase = desfasagem(t_cont_clipped, controle_clipped, t_rate_clipped, rate_clipped)
    return ganho_dB, fase

def figuras(parametros: Parametros):
    ulog = read_ulog(parametros.file_name)
    t_controle, controle, t_p_rate, p_rate = coleta_dados_p_rate(ulog)
    t0_clip, t1_clip, _, _ = offboard_time_analysis(ulog)
    t0_clip = t0_clip + parametros.time_wait
    t_cont_clipped, controle_clipped, t_rate_clipped, rate_clipped = filtra_recorta_dados(t_controle, controle,
                                                                                          t_p_rate, p_rate,
                                                                                          parametros.cutoff_freq_1,
                                                                                          parametros.cutoff_freq_2,
                                                                                          parametros.filt_ordem,
                                                                                          t0_clip,
                                                                                          t1_clip)
    # Detectar pontos críticos
    tc_pc_min, c_pc_min, tc_pc_max, c_pc_max = pontos_criticos(t_cont_clipped, controle_clipped)
    tp_pc_min, p_pc_min, tp_pc_max, p_pc_max = pontos_criticos(t_rate_clipped, rate_clipped)
    # Determinação dos cruzamentos por zero
    tc_cross, tc_cross_dir = cruzamento_zero(t_cont_clipped, controle_clipped)
    tp_cross, tp_cross_dir = cruzamento_zero(t_rate_clipped, rate_clipped)

    # Plotando os dados recortados
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.set_title(f"{parametros.file_name}, w0 = {parametros.w0:.2f} (rad/s), f0 = {parametros.f0:.2f} (Hz)")
    ax1.plot(t_controle, controle, label='controle')
    ax1.plot(tc_pc_min, c_pc_min, 'ok', label='pontos críticos min')
    ax1.plot(tc_pc_max, c_pc_max, 'or', label='pontos críticos max')
    ax1.plot(t_cont_clipped, controle_clipped, label='controle (filt, clipped)')
    for i in range(len(tc_cross)):
        if tc_cross_dir[i] > 0:
            ax1.plot(tc_cross[i], 0.0, 'xg')
        elif tc_cross_dir[i] < 0:
            ax1.plot(tc_cross[i], 0.0, 'xb')
    # ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Controle')
    ax1.set_xlim(t0_clip, t1_clip)
    ax1.set_ylim(1.1 * min(controle_clipped), 1.1 * max(controle_clipped))
    ax1.legend()
    ax1.grid()
    #
    ax2.plot(t_p_rate, p_rate, label='p')
    ax2.plot(tp_pc_min, p_pc_min, 'ok', label='pontos críticos min')
    ax2.plot(tp_pc_max, p_pc_max, 'or', label='pontos críticos max')
    ax2.plot(t_rate_clipped, rate_clipped, label='p (filt, clipped)')
    for i in range(len(tp_cross)):
        if tp_cross_dir[i] > 0:
            ax2.plot(tp_cross[i], 0.0, 'xg')
        elif tp_cross_dir[i] < 0:
            ax2.plot(tp_cross[i], 0.0, 'xb')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Velocidade angular')
    ax2.set_xlim(t0_clip, t1_clip)
    ax2.set_ylim(1.1 * min(rate_clipped), 1.1 * max(rate_clipped))
    ax2.legend()
    ax2.grid()
    plt.show()

def analise_ulog(parametros: Parametros):
    """
    Analyzes an ulog file to extract control and rate data, process it using filtering and clipping,
    and computes Bode plot parameters. Optionally generates plots for the processed data.

    Args:
        parametros (Parametros): A configuration and input object that contains details
            such as file name of the ulog, time adjustments, cutoff frequencies, filter order,
            and a flag indicating whether to create plots.

    Returns:
        Tuple[float, float]: A tuple containing:
            - ganho_p_dB: The gain in dB computed at specified points of the control and rate data.
            - fase: The phase in degrees corresponding to the computed gain in the Bode analysis.
    """
    ulog = read_ulog(parametros.file_name)
    # obtem os dados a partir do objeto ulog
    t_controle, controle, t_p_rate, p_rate = coleta_dados_p_rate(ulog)
    # inicio e fim do offboard mode
    t0_clip, t1_clip, _, _ = offboard_time_analysis(ulog)
    t0_clip = t0_clip + parametros.time_wait
    # filtra e clipa os dados
    t_cont_clipped, controle_clipped, t_p_rate_clipped, p_rate_clipped = filtra_recorta_dados(t_controle, controle,
                                                                                              t_p_rate, p_rate,
                                                                                              parametros.cutoff_freq_1,
                                                                                              parametros.cutoff_freq_2,
                                                                                              parametros.filt_ordem,
                                                                                              t0_clip,
                                                                                              t1_clip)
    ganho_p_dB, fase = bode_point(t_cont_clipped, controle_clipped, t_p_rate_clipped, p_rate_clipped)
    if parametros.plot_fig:
        figuras(parametros)
    return ganho_p_dB, fase

def main ():
    """
    Main function to process a set of ulog data files, analyze them, and save the results
    to a text file. The function performs the following steps:
    1. Defines input data for multiple ulog files, including file name, frequency parameters,
       and cutoff frequencies.
    2. Iteratively processes each ulog file using defined parameters, extracting gain and
       phase data using external tools.
    3. Aggregates and stores results from the analysis (frequency, gain, and phase).
    4. Saves the aggregated results to a text file in a formatted structure.

    Raises:
        This function does not directly raise any specific exceptions, but external
        tools or libraries used within (e.g., numpy, custom classes/functions) could
        raise errors if inputs are invalid or if processing fails.
    """
    # ulog_dados = [[path+file_name, w0, cutoff_freq_1, cutoff_freq_2],...]
    ulogs_dados = [
        ['ulogs/log_0_2025-1-7-11-01-56.ulg', 0.1, 20*0.1/(2*np.pi), 15*0.1/(2*np.pi)],
        ['ulogs/log_0_2025-1-7-11-27-03.ulg', 0.2, 20*0.2/(2*np.pi), 15*0.2/(2*np.pi)],
        ['ulogs/log_0_2025-1-7-11-34-39.ulg', 0.4, 15*0.4/(2*np.pi), 15*0.4/(2*np.pi)],
        ['ulogs/log_1_2025-1-7-11-39-49.ulg', 0.8, 15*0.8/(2*np.pi), 15*0.8/(2*np.pi)],
        #['ulogs/log_3_2025-1-7-11-46-35.ulg', 1.6, 2*1.6/(2*np.pi), 15*1.6/(2*np.pi)], # need to collect again with higher amplitud
    ]

    ganho_p_dB = []
    fase = []
    w0s = []
    for i in range(len(ulogs_dados)):
        file_name = ulogs_dados[i][0]
        w0 = ulogs_dados[i][1]
        cutoff_freq_1 = ulogs_dados[i][2]
        cutoff_freq_2 = ulogs_dados[i][3]
        p_param = Parametros(file_name,
                             w0,
                             cutoff_freq_1,
                             cutoff_freq_2,
                             10,
                             True
                             )
        ganho_p_dB_i, fase_i = analise_ulog(p_param)
        ganho_p_dB.append(ganho_p_dB_i)
        fase.append(fase_i)
        w0s.append(w0)

    dados = [w0s, ganho_p_dB, fase]

    # Salvar no arquivo txt
    np.savetxt('dados/dados_p_sins.txt', np.column_stack(dados), header="w0, ganho_p_dB, fase",
                   fmt="%.6f")  # Ajuste 'fmt' para precisar do número de casas decimais desejado
    return

if __name__ == "__main__":
    main()
