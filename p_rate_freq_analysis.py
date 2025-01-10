import matplotlib.pyplot as plt
import numpy as np
import myfunctions as mf

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
        p_param = mf.Parametros(file_name,
                             w0,
                             cutoff_freq_1,
                             cutoff_freq_2,
                             10,
                             True
                             )
        ganho_p_dB_i, fase_i = mf.analise_ulog(p_param)
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
