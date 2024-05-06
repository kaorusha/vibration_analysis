import numpy as np
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import emd
import memd.MEMD_all
from itertools import islice

def read_data(file_name):
    workbook = openpyxl.load_workbook(file_name, read_only=True, data_only=True, keep_links=False)
    # print("There are " + str(len(workbook.sheetnames)) + " sheets in this workbook")
    sheet_name = workbook.sheetnames[0]
    record_name = workbook[sheet_name]["B5"].value
    data = workbook[sheet_name].values
    data = list(data)[13:]
    cols = data[0][1:4]
    idx = [r[0] for r in data[1:]]
    data = (islice(r, 1, 4) for r in data[1:])
    df = pd.DataFrame(data, index=idx, columns=cols)
    print("df shape = (%s , %s)" % df.shape)
    return df, record_name

color = {
    'blue': (0, 0.4470, 0.7410),
    'orange': (0.8500, 0.3250, 0.0980),
    'red': (0.6350, 0.0780, 0.1840),
    'violet': (0.4940, 0.1840, 0.5560),
    'green': (0.4660, 0.6740, 0.1880),
    'cyan': (0.3010, 0.7450, 0.9330)
}

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
    :param imfs: imfs.shape = (NUM_OF_IMFS,NUM_OF_VARIANT,LENGTH_OF_DATA)
    :param print_imf: imfs higher than print_imf is summed up as residual
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

def memd_demo():
    data_combined, t = synthetic_data()
    #plot_data(data_combined, t, color_list=[color['blue'], color['orange'], color['green']])
    k = 64
    stopCrit = [0.075, 0.75, 0.075]
    x = np.array(data_combined)
    imfs = memd.MEMD_all.memd(x, k, 'stop', stopCrit)
    # print(imfs.shape) = (7,3,2500)
    plot_imfs(imfs, t, stopTime_plot = 2, color_list=[color['blue'], color['orange'], color['green']])

def fft(df:pd.DataFrame, title:str):
    acc = df[df.columns[2]]
    sp = np.fft.rfft(acc, norm='backward')
    freq = np.fft.rfftfreq(n=len(acc), d=1/48000)
    plt.plot(freq[500:50000], np.abs(sp[500:50000]))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(title + ' ' + df.columns[2])
    plt.show()

def fft_test():
    N = 500
    T = 1.0 / 600.0
    x = np.linspace(0.0, N * T, N)
    y = np.sin(60.0 * 2.0 * np.pi * x) + 0.5 * np.sin(90.0 * 2.0 * np.pi * x)
    y_f = np.fft.rfft(y, norm='forward')
    x_f = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(x_f, np.abs(y_f[:N // 2]))
    plt.show()

def test_emd():
    t = df.index[:48000]
    acc_all = df.transpose().to_numpy()
    #imf = emd.sift.sift(acc.to_numpy())
    stopCrit = [0.075, 0.75, 0.075]
    imfs = memd.MEMD_all.memd(acc_all[:,:48000], 64, 'stop', stopCrit)
    print(imfs.shape) # (18, 3, 96000)
    plot_imfs(imfs, t, stopTime_plot = 1, color_list=[color['blue'], color['orange'], color['green']], print_imf=imfs.shape[0]-1)

file_name = "d:\\cindy_hsieh\\My Documents\\project\\vibration_analysis\\test_data\\raw_data_20240308\\richard\\20240222Level_vs_Time.xlsx"
df, title = read_data(file_name)
#fft(df, title)
#test_emd()
#imf_x = imf[:,0,:] #imfs corresponding to 1st component
#imf_y = imf[:,1,:] #imfs corresponding to 2nd component
#imf_z = imf[:,2,:] #imfs corresponding to 3rd component
#axes_x = emd.plotting.plot_imfs(imf_x.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#axes_y = emd.plotting.plot_imfs(imf_y.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#axes_z = emd.plotting.plot_imfs(imf_z.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#plt.close("all")
