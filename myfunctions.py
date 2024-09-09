

# Função para ler o arquivo .ulg e extrair dados

"""
Reads a .ulg file and returns the ULog object.

Args:
    file_path (str): The path to the .ulg file.

Returns:
    ULog: The ULog object representing the .ulg file.
"""

"""
Lists all field_data of a ULog object.

Args:
    ulog (ULog): The ULog object.

Prints:
    str: The topic name.
    str: The field name.
"""

"""
Plots the data from a specific topic and field of a ULog object.

Args:
    ulog (ULog): The ULog object.
    topic_name (str): The name of the topic.
    field_name (str): The name of the field.

Displays:
    A plot of the field data over time.
"""

"""
Retrieves the data from a specific topic and field of a ULog object.

Args:
    ulog (ULog): The ULog object.
    topic_name (str): The name of the topic.
    field_name (str): The name of the field.

Returns:
    numpy.ndarray: The timestamps.
    numpy.ndarray: The field data.
"""

"""
Designs a Butterworth lowpass filter.

Args:
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency.
    order (int, optional): The order of the filter. Defaults to 5.

Returns:
    numpy.ndarray: The numerator coefficients of the filter.
    numpy.ndarray: The denominator coefficients of the filter.
"""

"""
Applies a Butterworth lowpass filter to the input data.

Args:
    data (numpy.ndarray): The input data.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency.
    order (int, optional): The order of the filter. Defaults to 5.

Returns:
    numpy.ndarray: The filtered data.
"""
from pyulog import ULog
from scipy.signal import butter, lfilter
from myfunctions import *
import numpy as np
import math

def read_ulog(file_path):
    ulog = ULog(file_path)
    return ulog

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