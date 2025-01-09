import memd.MEMD_all
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

color = {
    'blue': (0, 0.4470, 0.7410),
    'orange': (0.8500, 0.3250, 0.0980),
    'red': (0.6350, 0.0780, 0.1840),
    'violet': (0.4940, 0.1840, 0.5560),
    'green': (0.4660, 0.6740, 0.1880),
    'cyan': (0.3010, 0.7450, 0.9330)
}

def test_emd(df:pd.DataFrame):
    t = df.index[:48000]
    acc_all = df.transpose().to_numpy()
    #imf = emd.sift.sift(acc.to_numpy())
    stopCrit = [0.075, 0.75, 0.075]
    imfs = memd.MEMD_all.memd(acc_all[:,:48000], 64, 'stop', stopCrit)
    print(imfs.shape) # (18, 3, 96000)
    plot_imfs(imfs, t, stopTime_plot = 1, color_list=[color['blue'], color['orange'], color['green']], print_imf=imfs.shape[0]-1)

def test_emd(df:pd.DataFrame):
    t = df.index[:48000]
    acc_all = df.transpose().to_numpy()
    #imf = emd.sift.sift(acc.to_numpy())
    stopCrit = [0.075, 0.75, 0.075]
    imfs = memd.MEMD_all.memd(acc_all[:,:48000], 64, 'stop', stopCrit)
    print(imfs.shape) # (18, 3, 96000)
    plot_imfs(imfs, t, stopTime_plot = 1, color_list=[color['blue'], color['orange'], color['green']], print_imf=imfs.shape[0]-1)

def synthetic_data():
    fs = 500
    dt = 1/fs
    stopTime = 5
    t = np.linspace(0, stopTime, int(fs * stopTime), endpoint=False)
    F_1 = 50
    F_2 = 20
    F_3 = 5
    A_1 = 4
    A_2 = 5
    A_3 = 3
    data_1 = A_1 + A_1*np.sin(2*np.pi*F_1*t) + A_1*np.sin(2*np.pi*F_2*t)
    data_2 =       A_2*np.sin(2*np.pi*F_1*t) + A_2*np.sin(2*np.pi*F_3*t)
    data_3 = A_3 + A_3*np.sin(2*np.pi*F_1*t) + A_3*np.sin(2*np.pi*F_2*t) + A_3*np.sin(2*np.pi*F_3*t)
    data_combined = [data_1, data_2, data_3]
    return data_combined, t

def plot_data(data_combined, t, color_list, stopTime_plot = 2):
    fig, axs = plt.subplots(len(data_combined),1)
    ylables = ['$g_{i}(t)$' for i in range(len(data_combined))]
    lable_font = {'fontname': 'Times New Roman', 'style':'italic'}
    for ax, color, data, ylable in zip(axs, color_list, data_combined, ylables):
        ax.plot(t, data, c=color)
        ax.set_xlim(0, stopTime_plot)
        ax.set_ylabel(ylable, **lable_font)
        ax.set_xlabel('$t$', **lable_font)
    plt.show()

def plot_imfs(imfs: np.ndarray, t, stopTime_plot, color_list:list, print_imf = 3):
    """
    Parameters
    ----------
    imfs : imfs.shape = (NUM_OF_IMFS,NUM_OF_VARIANT,LENGTH_OF_DATA)
    print_imf : imfs higher than print_imf is summed up as residual
    """
    fig, axs = plt.subplots(print_imf + 1, imfs.shape[1])
    lable_font = {'fontname': 'Times New Roman', 'style':'italic'}
    dd = sum(imfs[print_imf:,:,:])
    
    for i in range(print_imf + 1):
        for j in range(imfs.shape[1]):
            if i == print_imf:            
                axs[i][j].plot(t, dd[j], c=color_list[j])
                axs[i][j].set_xlabel('$t$', **lable_font)
            else:
                axs[i][j].plot(t, imfs[i][j], c=color_list[j])
            axs[i][j].set_xlim(0, stopTime_plot)
            if i == 0:
                axs[i][j].set_title('$g_{%d}(t)$'%(j+1), **lable_font)
            if j == 0:
                label = 'res' if i == print_imf else 'IMF%d'%(i+1)
                axs[i][j].set_ylabel(label, **lable_font)
    plt.show()
