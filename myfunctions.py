

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