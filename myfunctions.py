# Funções para ler o arquivo .ulg e extrair dados
from pyulog import ULog
from scipy.signal import butter, lfilter, filtfilt
from myfunctions import *
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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

def recorta_dados_2(t, x, t0, t1):
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
    t_cont_clipped, controle_clipped = recorta_dados_2(t_controle, controle_filt, t0_clip, t1_clip)
    t_rate_clipped, rate_clipped = recorta_dados_2(t_rate, rate_filt, t0_clip, t1_clip)
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


# Dicionário de nomes dos modos de voo para um multicopter
flight_mode_names_multicopter = {
    0: 'Manual',
    1: 'Altitude',
    2: 'Position',
    3: 'Auto: Mission',
    4: 'Auto: Loiter',
    5: 'Auto: RTL',
    6: 'Acro',
    7: 'Stabilized',
    8: 'Rattitude',
    9: 'Auto: Takeoff',
    10: 'Auto: Land',
    11: 'Auto: Follow Target',
    12: 'Auto: Precision Land',
    13: 'Auto: VTOL Takeoff',
    14: 'Auto: VTOL Land',
    15: 'Auto: VTOL Transition to FW',
    16: 'Auto: VTOL Transition to MC',
    17: 'Offboard'
}

def fft_one_sided(t, y, plot=False, label='Magnitude'):
    """
    Compute the one-sided Fast Fourier Transform (FFT) of a signal.
    Parameters:
    t (array-like): Time array.
    y (array-like): Signal array corresponding to the time array `t`.
    Returns:
    tuple: A tuple containing:
        - freq_one_sided (array-like): One-sided frequency array.
        - fft_y_one_sided (array-like): One-sided FFT of the signal `y`.
    """
    dt = t[1] - t[0]
    fft_y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), dt)
    fft_y_one_sided = fft_y[:len(fft_y)//2]
    freq_one_sided = freq[:len(freq)//2]
    
    if plot:
        plt.figure(figsize=(10, 10))
        plt.subplot(2,1,1)
        plt.plot(freq_one_sided, np.abs(fft_y_one_sided), label=label)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('One-Sided FFT')
        plt.legend()
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(freq_one_sided, np.unwrap(np.angle(fft_y_one_sided)), label=label)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (Radians)')
        plt.grid()
        plt.legend()
        plt.show()
        
    return freq_one_sided, fft_y_one_sided

def plot3x1(x1, y1, y1_label, x2, y2, y2_label, x3, y3, y3_label):
    """
    Plots three subplots in a single figure, each with its own x and y data.
    Parameters:
    x1 (array-like): Data for the x-axis of the first subplot.
    y1 (array-like): Data for the y-axis of the first subplot.
    y1_label (str): Label for the y-axis data of the first subplot.
    x2 (array-like): Data for the x-axis of the second subplot.
    y2 (array-like): Data for the y-axis of the second subplot.
    y2_label (str): Label for the y-axis data of the second subplot.
    x3 (array-like): Data for the x-axis of the third subplot.
    y3 (array-like): Data for the y-axis of the third subplot.
    y3_label (str): Label for the y-axis data of the third subplot.
    Returns:
    None
    This function creates a figure with three subplots arranged vertically.
    Each subplot shares the same x-axis (time in seconds) but has different y-axis data.
    The function also adds labels, legends, and grids to each subplot for better visualization.
    """
    
    # Criar uma figura com 3 subplots (3 linhas, 1 coluna)
    fig, axs = plt.subplots(3, 1, figsize=(9, 6))

    # Plotar os ângulos de Euler ao longo do tempo
    axs[0].plot(x1, y1, label=y1_label)
    axs[0].set_xticklabels([])
    axs[0].legend()
    axs[0].grid(True)

    # Plotar os comandos de controle ao longo do tempo
    axs[1].plot(x2, y2, label=y2_label)
    axs[1].set_xticklabels([])
    axs[1].legend()
    axs[1].grid(True)

    # Plotar as taxas de rotação ao longo do tempo
    axs[2].plot(x3, y3, label=y3_label)
    axs[2].set_xlabel('Time (segundos)')
    axs[2].legend()
    axs[2].grid(True)

    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()
    
def get_euler_taxa(file_path = 'ulogs/log_13_2024-9-16-08-21-12.ulg', angle='roll', taxa='p', sinal=[1,-1,-1,1]):
    """
    Process and plot Euler angles and control signals from a ULog file.
    This function reads a ULog file, extracts relevant flight data, processes
    quaternion attitudes into Euler angles, and generates control signals. It
    then plots the Euler angles, rotation rates, and control signals.
    Parameters:
    file_path (str): Path to the ULog file. Default is 'ulogs/log_13_2024-9-16-08-21-12.ulg'.
    angle (str): The Euler angle to extract ('roll', 'pitch', 'yaw'). Default is 'roll'.
    taxa (str): The rotation rate to extract ('p', 'q', 'r'). Default is 'p'.
    sinal (list): List of coefficients to generate the control signal. Default is [1, -1, -1, 1].
    Returns:
    tuple: Contains the following elements:
        - timestamps_clipped (list): Timestamps for the Euler angles.
        - roll (list): Extracted Euler angles.
        - timestamps_taxas_clipped (list): Timestamps for the rotation rates.
        - p_clipped (list): Extracted rotation rates.
        - timestamps_cont_clipped (list): Timestamps for the control signals.
        - control_roll (list): Generated control signals.
    """
    
    # Ler o arquivo .ulg
    ulog = read_ulog(file_path)

    # obtem os timestamps e os modos de voo
    change_timestamps, change_modes = get_flight_mode_changes(ulog)

    # obtem o timestamps do offboard e do rtl
    mode_timestamps = get_mode_timestamps(change_timestamps, change_modes)

    # coleta a atitude do veiculo em quaternios
    timestamps, q_d0, q_d1, q_d2, q_d3 = coleta_quaternion_atitude(ulog)

    # coleta os comandos de controle do veículo
    timestamps_cont, control0, control1, control2, control3 = coleta_controles(ulog)

    # Coletar os dados das taxas de rotação
    timestamps_taxas, p = coleta_taxas_rotacao(ulog, taxa=taxa)

    # recorta os dados
    timestamps_clipped, q_d0_clipped = recorta_dados(timestamps, q_d0, mode_timestamps)
    _, q_d1_clipped = recorta_dados(timestamps, q_d1, mode_timestamps)
    _, q_d2_clipped = recorta_dados(timestamps, q_d2, mode_timestamps)
    _, q_d3_clipped = recorta_dados(timestamps, q_d3, mode_timestamps)

    # Quaternions to Euler Angles
    roll = get_euler_angles_from_quat(q_d0_clipped, q_d1_clipped, q_d2_clipped, q_d3_clipped, angle=angle)

    #
    timestamps_cont_clipped, control0_clipped = recorta_dados(timestamps_cont, control0, mode_timestamps)
    _, control1_clipped = recorta_dados(timestamps_cont, control1, mode_timestamps)
    _, control2_clipped = recorta_dados(timestamps_cont, control2, mode_timestamps)
    _, control3_clipped = recorta_dados(timestamps_cont, control3, mode_timestamps)

    # gerar o sinal de controle de roll
    control_roll = sinal[0]*control0_clipped + sinal[1]*control1_clipped + sinal[2]*control2_clipped + sinal[3]*control3_clipped

    # Recortar os dados das taxas de rotação
    timestamps_taxas_clipped, p_clipped = recorta_dados(timestamps_taxas, p, mode_timestamps)
    
    plot3x1(x1=timestamps_clipped, y1=roll, y1_label=angle, x2=timestamps_taxas_clipped, y2=p_clipped, y2_label=taxa, x3=timestamps_cont_clipped, y3=control_roll, y3_label='control')
    
    return timestamps_clipped, roll, timestamps_taxas_clipped, p_clipped, timestamps_cont_clipped, control_roll  

def coleta_taxas_rotacao(ulog, taxa='p'):
    if taxa == 'p':
        timestamps, p = get_ulog_data(ulog, 'vehicle_angular_velocity', 'xyz[0]')
        return timestamps, p
    elif taxa == 'q':
        timestamps, q = get_ulog_data(ulog, 'vehicle_angular_velocity', 'xyz[1]')
        return timestamps, q
    elif taxa == 'r':
        timestamps, r = get_ulog_data(ulog, 'vehicle_angular_velocity', 'xyz[2]')
        return timestamps, r
    else:
        raise ValueError('A taxa deve ser "p", "q" ou "r".')
    return taxa

def get_euler_angles_from_quat(q_d0, q_d1, q_d2, q_d3, angle='roll'):
    # Converter todos os quaternions para ângulos de Euler
    euler_angles = np.array([quaternion_to_euler(q0, q1, q2, q3) for q0, q1, q2, q3 in zip(q_d0, q_d1, q_d2, q_d3)])
    if angle == 'roll':
        return euler_angles[:, 0]
    elif angle == 'pitch':
        return euler_angles[:, 1]
    elif angle == 'yaw':
        return euler_angles[:, 2]
    else:
        raise ValueError('O ângulo deve ser "roll", "pitch" ou "yaw".')
    return angle

def coleta_controles(ulog):
    timestamps, control0 = get_ulog_data(ulog, 'actuator_motors', 'control[0]')
    _, control1 = get_ulog_data(ulog, 'actuator_motors', 'control[1]')
    _, control2 = get_ulog_data(ulog, 'actuator_motors', 'control[2]')
    _, control3 = get_ulog_data(ulog, 'actuator_motors', 'control[3]')
    return timestamps, control0, control1, control2, control3

def recorta_dados(timestamps, q_d0, mode_timestamps):
    """
    Recorta os vetores de timestamps e q_d0 com início no instante Offboard e final no instante RTL.

    :param timestamps: numpy.ndarray, vetor de timestamps
    :param q_d0: numpy.ndarray, vetor de quaternions q_d0
    :param mode_timestamps: dict, dicionário com os timestamps dos modos de voo
    :return: tuple, vetores recortados de timestamps e q_d0
    """
    offboard_time = mode_timestamps['Offboard'][0]
    rtl_time = mode_timestamps['RTL'][0]

    # Encontrar os índices correspondentes aos tempos de Offboard e RTL
    start_index = np.searchsorted(timestamps, offboard_time, side='left')
    end_index = np.searchsorted(timestamps, rtl_time, side='right')

    # Recortar os vetores
    recortados_timestamps = timestamps[start_index:end_index]
    recortados_q_d0 = q_d0[start_index:end_index]

    return recortados_timestamps, recortados_q_d0

# coletar dados do sinal de entrada
def coleta_quaternion_atitude(ulog):
    timestamps, q_d0 = get_ulog_data(ulog, 'vehicle_attitude_setpoint', 'q_d[0]')
    _, q_d1 = get_ulog_data(ulog, 'vehicle_attitude_setpoint', 'q_d[1]')
    _, q_d2 = get_ulog_data(ulog, 'vehicle_attitude_setpoint', 'q_d[2]')
    _, q_d3 = get_ulog_data(ulog, 'vehicle_attitude_setpoint', 'q_d[3]')
    return timestamps, q_d0, q_d1, q_d2, q_d3

def get_mode_timestamps(change_timestamps, change_modes, mode_offboard=14, mode_rtl=5):
    """
    Retorna os timestamps quando os modos Offboard e RTL começam.

    Parameters:
    change_timestamps (numpy.ndarray): Array de timestamps das mudanças de modo.
    change_modes (numpy.ndarray): Array de modos correspondentes aos timestamps.
    mode_offboard (int): Código do modo Offboard. Default é 14.
    mode_rtl (int): Código do modo RTL. Default é 5.

    Returns:
    dict: Dicionário com os timestamps dos modos Offboard e RTL.
    """
    offboard_timestamps = change_timestamps[change_modes == mode_offboard]
    rtl_timestamps = change_timestamps[change_modes == mode_rtl]
    
    return {
        'Offboard': offboard_timestamps,
        'RTL': rtl_timestamps
    }

def get_flight_mode_changes(ulog, topic_name='vehicle_status', field_name='nav_state_user_intention'):
    """
    Função para obter os instantes nos quais os modos de voo mudam e imprimir os nomes dos modos de voo.
    
    Parâmetros:
    ulog: objeto ULog
    topic_name: nome do tópico que contém o estado de navegação (default: 'vehicle_status')
    field_name: nome do campo que contém o estado de navegação (default: 'nav_state')
    
    Retorna:
    timestamps: lista de instantes nos quais os modos de voo mudam
    flight_modes: lista dos modos de voo correspondentes aos instantes
    """

    # Coletar dados do tópico
    timestamps, flight_modes = get_ulog_data(ulog, topic_name, field_name)
    
    # Identificar mudanças nos modos de voo
    changes = np.where(np.diff(flight_modes) != 0)[0] + 1
    
    # Obter os instantes e modos de voo correspondentes às mudanças
    change_timestamps = timestamps[changes]
    change_modes = flight_modes[changes]
    
    return change_timestamps, change_modes

def offboard_time_analysis(ulog):
    """
    Analyzes the offboard control mode activation and deactivation times from the ulog data.
    Parameters:
    ulog (object): The ulog object containing the flight log data.
    Returns:
    tuple: A tuple containing:
        - offboard_flag_up_time (numpy.ndarray): Timestamps when the offboard mode was activated.
        - offboard_flag_down_time (numpy.ndarray): Timestamps when the offboard mode was deactivated.
        - offboard_flag_timestamp (numpy.ndarray): All timestamps corresponding to the offboard flag data.
        - offboard_flag_data (numpy.ndarray): The offboard flag data indicating whether the offboard mode is enabled.
    """
    # vehicle_control_mode
    offboard_flag = get_ulog_data(ulog, 'vehicle_control_mode', 'flag_control_offboard_enabled')
    offboard_flag_timestamp = offboard_flag[0]
    offboard_flag_data = offboard_flag[1]
    # Calcula a derivada do offboard_flag_data
    offboard_flag_diff = np.diff(offboard_flag_data)
    # Encontra os indices onde o offboard_flag_data muda
    offboard_flag_up_index = np.where(offboard_flag_diff==1)
    offboard_flag_down_index = np.where(offboard_flag_diff==-1)
    # Tempo em que o offboard mode foi ativado e desativado
    offboard_flag_up_time = offboard_flag_timestamp[offboard_flag_up_index]
    offboard_flag_down_time = offboard_flag_timestamp[offboard_flag_down_index]
    return offboard_flag_up_time, offboard_flag_down_time, offboard_flag_timestamp, offboard_flag_data

def quaternion_to_euler(q0, q1, q2, q3):
    """
    Converte um quaternion em ângulos de Euler.
    
    Parâmetros:
    q0, q1, q2, q3: Componentes do quaternion
    
    Retorna:
    roll, pitch, yaw: Ângulos de Euler em radianos
    """
    r = R.from_quat([q1, q2, q3, q0])  # scipy usa a ordem [x, y, z, w]
    euler = r.as_euler('xyz', degrees=False)
    return euler

def read_ulog(file_path):
    ulog = ULog(file_path)
    return ulog

def list_topics(ulog):
    for dataset in ulog.data_list:
        print(dataset.name)

def list_fields(ulog, topic_name):
    dataset = ulog.get_dataset(topic_name)
    for field_name in dataset.data.keys():
        if field_name != 'timestamp':  # Ignorar o campo de timestamp
            print(field_name)

# Função para listar todos os field_data de um arquivo .ulg
def list_all_fields(ulog):
    for dataset in ulog.data_list:
        topic_name = dataset.name
        print(f'Tópico: {topic_name}')
        for field_name in dataset.data.keys():
            if field_name != 'timestamp':  # Ignorar o campo de timestamp
                print(f'  Campo: {field_name}')

# Função para plotar os dados
def plot_ulog_data(ulog, topic_name, field_name):
    data = ulog.get_dataset(topic_name).data
    timestamps = data['timestamp'] / 1e6  # Convertendo de microssegundos para segundos
    field_data = data[field_name]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, field_data, label=field_name)
    plt.xlabel('Tempo (s)')
    plt.ylabel(field_name)
    plt.title(f'{field_name} ao longo do tempo')
    plt.legend()
    plt.grid()
    plt.show()

def get_ulog_data(ulog, topic_name, field_name):
    data = ulog.get_dataset(topic_name).data
    timestamps = data['timestamp'] / 1e6  # Convertendo de microssegundos para segundos
    field_data = data[field_name]

    return timestamps, field_data

# Filtros
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Cria um filtro Butterworth passa-faixa.

    :param lowcut: Frequência de corte inferior (Hz).
    :param highcut: Frequência de corte superior (Hz).
    :param fs: Frequência de amostragem (Hz).
    :param order: Ordem do filtro.
    :return: Coeficientes do filtro (b, a).
    """
    nyquist = fs / 2.0  # Frequência de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica um filtro Butterworth passa-faixa a um sinal.

    :param data: Sinal de entrada.
    :param lowcut: Frequência de corte inferior (Hz).
    :param highcut: Frequência de corte superior (Hz).
    :param fs: Frequência de amostragem (Hz).
    :param order: Ordem do filtro.
    :return: Sinal filtrado.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)  # Filtragem bidirecional para evitar atraso de fase
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def rk4(f, x0, t0, tf, h):
    # Implementa o algoritmo Runge-Kutta de 4ta ordem
    # com passo de integração fixo.
    # dotx = f(t,x)
    # x0 = numpy.array([x1,...,xn]), vetor de dimensão n
    # t0 : tempo inicial, escalar não negativo
    # tf : tempo final, escalar não negativo
    # h : passo de integração, é um escalar positivo
    # as saídas são:
    # t : o vetor tempo,
    # x : o vetor de estados
    #
    from numpy import zeros, absolute, floor, minimum, abs, flip, fliplr, flipud
    import warnings
    #
    N = absolute(floor((tf-t0)/h)).astype(int)
    x = zeros((N+1, x0.size))
    t = zeros(N+1)
    x[0, :] = x0
    # verification
    if (tf < t0):
        warnings.warn("Backwards integration requested", Warning)
        h = -abs(h)
        back_flag = True
    else:
        back_flag = False
    if (tf == t0):
        raise ValueError("t0 equals tf")
    if (abs(tf-t0) <= abs(h)):
        raise ValueError("Integration step h is too long")
    t[0] = t0
    for i in range(0, N):
        k1 = f(t[i], x[i])
        k2 = f(t[i]+h/2.0, x[i]+(h*k1)/2.0)
        k3 = f(t[i]+h/2.0, x[i]+(h*k2)/2.0)
        k4 = f(t[i]+h, x[i]+h*k3)
        x[i+1, :] = x[i, :]+(h/6.0)*(k1+2.0*k2+2.0*k3+k4)
        t[i+1] = t[i]+h
    if back_flag == True:
        t = flip(t)
        x = flipud(x)
    return t, x
#
def dyn_p(w, Ixx):
    '''
    Dinâmica linear da taxa do ângulo de rolagem
    w : frequência angular em rad/s
    Ixx : momento de inércia em kg.m^2
    
    Retorna a função de transferência
    '''
    s = complex(0,w)
    return 1 / (Ixx*s)

def delay_p(w, tau):
    """
    Calculates the delay transfer function for a given frequency and time constant.
    Parameters:
    - w (float): The frequency value in rad/s.
    - tau (float): The time constant value in seconds.
    Returns:
    - complex: The complex exponential value representing the delay transfer function.
    """
    s = complex(0,w)
    return np.exp(-tau*s)

def compensador(w, a, T, k):
    """
    Calculates the compensator transfer function.
    Parameters:
    w (float): Frequency value in rad/s.
    a (float): Coefficient value.
    T (float): Time constant value.
    k (float): Gain value.
    Returns:
    complex: The transfer function of the compensator.
    """
    s = complex(0,w)
    return k*(T*s + 1)/(a*T*s + 1)

def G(w, Ixx, tau, alpha, T, k):
    """
    Calculates the transfer function G(s) given the parameters.
    Parameters:
    w (float): The frequency value in rad/s.
    Ixx (float): The moment of inertia value.
    tau (float): The time delay value in seconds.
    a (float): The coefficient value.
    T (float): The time constant value.
    k (float): The gain value.
    Returns:
    complex: The value of G(s) at the given frequency.
    """
    return dyn_p(w, Ixx)*delay_p(w, tau)*compensador(w, alpha ,T, k)    

def G_bode(w, G):
    """
    Calculates the magnitude (in dB) and phase (in degrees) of a transfer function G at given frequencies.

    Parameters:
    w (array-like): Array of frequencies in rad/s at which to evaluate the transfer function.
    G (callable): Transfer function G(w) that takes a frequency w as input and returns a complex value.

    Returns:
    mod_G_dB (array-like): Array of magnitudes of G(w) in decibels.
    G_fase (array-like): Array of phases of G(w) in degrees.
    """
    mod_G_dB = np.zeros(len(w))
    G_fase = np.zeros(len(w))
    for i in range(len(w)):
        mod_G_dB[i] = 20*np.log10(abs(G(w[i])))
        G_fase[i] = np.angle(G(w[i]))*180/np.pi
    return mod_G_dB, np.unwrap(G_fase)
    
def calcula_param_comp_avan(fm, Gc, phim):
    """
    Calculates the advance compensator parameters a, T, and k based on the 
    given inputs.
    Parameters:
    fm (float): The frequency in Hz.
    Gc (float): The gain in dB.
    phim (float): The phase in degrees.
    Returns:
    tuple: A tuple containing the calculated values of a, T, and k.
        - a (float): The calculated value of a.
        - T (float): The calculated value of T.
        - k (float): The calculated value of k.
    """
    
    a = (1-np.sin(phim*np.pi/180))/(1+np.sin(phim*np.pi/180))
    T = 1/(fm*2*np.pi*np.sqrt(a))
    k = math.pow(10,Gc/20)*np.sqrt(a)
    return a, T, k