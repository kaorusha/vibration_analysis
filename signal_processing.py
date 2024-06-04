from typing import Any
import librosa
import matplotlib.axes
from matplotlib.pylab import det
from scipy import signal
import numpy as np
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import emd
import memd.MEMD_all
from itertools import islice
import os

def workbook_to_dataframe(workbook:openpyxl.Workbook, sheet_name:str, channel:int):
    """
    workbook: transfer from head-acoustic data frame (abbreviated as file extension .hdf),
    with analysis type **Level vs. Time (Fast)**, which is the raw data of accelerometers.
    :param sheet_name: each sheet as one hdf file. The head-acoustic .hdf file name is at cell 'B5'.
    :param channel: input signal channel number, for example, the number of accelerometers.
    """
    record_name = workbook[sheet_name]["B5"].value
    data = workbook[sheet_name].values
    data = list(data)[13:]
    cols = data[0][1:channel+1]
    idx = [r[0] for r in data[1:]]
    data = (islice(r, 1, channel+1) for r in data[1:])
    df = pd.DataFrame(data, index=idx, columns=cols)
    print("sheet: " + sheet_name + "\t df shape = (%s , %s)" % df.shape)
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

def largestpowerof2(n:int):
    '''
    return a number which is the largest power of 2 and less than n
    '''
    while(n & (n - 1)):
        n &= (n - 1)
    return n

def butter_highpass(input, t, cutoff, fs, order = 5, axis = 0, visualize = False):
    sos = signal.butter(order, cutoff, btype='highpass', fs=fs, output='sos')
    output = signal.sosfilt(sos, input, axis=axis)
    if visualize:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, layout="tight")
        ax1.plot(t,input)
        ax1.set_title('input signal')
        ax2.plot(t, output)
        ax2.set_title('After %d hz high-pass filter'%cutoff)
        ax2.set_xlabel('Time [seconds]')
    return output

def fft(df:pd.DataFrame, fs = 1, nperseq=8192, noverlap=8192//2, axis=1):
    """
    do FFT with hanning window frame
    """
    frames = librosa.util.frame(df, frame_length=nperseq, hop_length=int(nperseq-noverlap), axis=0)
    window = np.hanning(nperseq)
    windowed_frames = np.empty(frames.shape)
    for col in range(frames.shape[-1]):
        np.multiply(window, frames[:,:,col], out=windowed_frames[:,:,col])
    sp = np.fft.rfft(windowed_frames, n=nperseq, axis=axis, norm='backward')
    freq = np.fft.rfftfreq(n=nperseq, d=1./fs)
    abs_sp = np.mean(np.abs(sp), axis=0)
    return freq, abs_sp

def get_fft(df: pd.DataFrame, frame_len=8192, fs = 48000, overlap = 0.75):
    '''
    return fft as dataframe type
    :param fs: sampling frequency
    ----
    signal processing step:
    1. subtract dc bias from accelerometer data
    2. high pass filter
    3. do FFT with hanning window frame
    4. get mean of FFT spectrum
    '''
    # subtract dc bias from acc data
    detrend_df = df - np.mean(df.to_numpy(), axis=0)
    # use high-pass filter to remove dc 
    filtered_df = butter_highpass(detrend_df, df.index, 60, fs, 2)
    freq, sp = fft(filtered_df, fs=fs, nperseq=frame_len, noverlap=frame_len*overlap)
    df_fft = pd.DataFrame(sp, columns=df.columns)
    df_fft['Frequency (Hz)'] = freq
    df_fft = df_fft.set_index('Frequency (Hz)')
    return df_fft

def annotate_peak(a: Any, freq: Any, ax: matplotlib.axes.Axes = None, prominence:Any|None = None):    
    """
    analysis the peak and add annotation on the graph
    :param a: 1d array spectrum absolute value
    :param freq: 1d array frequency
    """
    peaks, dic = signal.find_peaks(a, prominence=prominence)
    if ax != None:
        for idx in peaks:
            ax.annotate('%d'%freq[idx], 
                        xy=(freq[idx], a[idx]), rotation=45, xycoords='data',
                        xytext=(0, 30), textcoords='offset pixels',
                        arrowprops=dict(facecolor='blue', arrowstyle="->", connectionstyle="arc3"))
    return peaks

def test_emd():
    t = df.index[:48000]
    acc_all = df.transpose().to_numpy()
    #imf = emd.sift.sift(acc.to_numpy())
    stopCrit = [0.075, 0.75, 0.075]
    imfs = memd.MEMD_all.memd(acc_all[:,:48000], 64, 'stop', stopCrit)
    print(imfs.shape) # (18, 3, 96000)
    plot_imfs(imfs, t, stopTime_plot = 1, color_list=[color['blue'], color['orange'], color['green']], print_imf=imfs.shape[0]-1)

def calc_rms(df: pd.DataFrame):
    rms = df.copy()**2
    rms = rms.mean()**0.5
    return rms

def stat_calc(df: pd.DataFrame):
    """
    calculate accelerate data peak, rms, crest factor and standard deviation
    """
    df_stats = pd.concat([df.abs().max(),calc_rms(df)],axis=1)
    df_stats.columns = ['Acceleration Peak (g)','Acceleration RMS (g)']
    df_stats['Crest Factor'] = df_stats['Acceleration Peak (g)'] / df_stats['Acceleration RMS (g)']
    df_stats['Standard Deviation (g)'] = df.std()
    df_stats.index.name = 'Data Set'
    return df_stats

def get_psd(df: pd.DataFrame, frame_len=8192, fs = 48000, overlap = 0.75):
    detrend_df = df - np.mean(df.to_numpy(), axis=0)
    filtered_df = butter_highpass(detrend_df, df.index, 60, fs, 2)
    f, psd = signal.welch(filtered_df, fs=fs, nperseg=frame_len, window='hann', noverlap=frame_len*overlap, axis=0)
    df_psd = pd.DataFrame(psd,columns=df.columns)
    df_psd.columns
    df_psd['Frequency (Hz)'] = f
    df_psd = df_psd.set_index('Frequency (Hz)')
    return df_psd

def save_bar_plot(name: Any, value:Any, plot_title:str, file_name:str, figsize:tuple = (10, 10), path_dir:str = './fig/'):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(layout="constrained", figsize=figsize)
    # Horizontal Bar Plot
    ax.barh(name, value)
    # Show top values 
    ax.invert_yaxis()
    # Add annotation to bars
    for i in range(len(value)):
        plt.text(value.iloc[i], i, 
                 str(round(value.iloc[i], 3)),
                 fontsize = 10, fontweight ='bold',verticalalignment="center",
                 color ='grey')
    
    # Add Plot Title
    ax.set_title(plot_title, loc ='left', fontsize=14)
    # save figure
    if not os.path.exists(path_dir):
        print('%s not exist, create new directory.'%path_dir)
        os.makedirs(path_dir)
    fig.savefig(path_dir+file_name, transparent=False, dpi=80, bbox_inches="tight")

def acc_processing(hdf_level_time_filename:str , 
                   state: bool = False, state_result_filename:str = 'state.xlsx', 
                   fft: bool = False, fft_result_filename:str = 'fft.xlsx',
                   psd: bool = False, psd_result_filename:str = 'psd.xlsx'):
    """
    read level vs time acoustic .hdf file, loop for each sheet, read as panda data frame and do selected processing, 
    save the result of from multiple sheet of raw acc data into one result excel sheet.

    :param state: whether use the data frame to calculate time domain standard deviation etc..
    :param fft: whether to calculate fast fourier transform
    :param psd: whether to calculate power spectral density
    """
    workbook = openpyxl.load_workbook(hdf_level_time_filename, read_only=True, data_only=True, keep_links=False)
    print("There are %d"%len(workbook.sheetnames) + " sheets in this workbook ( " + hdf_level_time_filename + " )")
    if state:
        df_all_stats = pd.DataFrame()
    if fft:
        df_all_fft = pd.DataFrame()
    if psd:
        df_all_psd = pd.DataFrame()
    for sheet in workbook.sheetnames:
        df, title = workbook_to_dataframe(workbook, sheet, 3)
        # rewrite column title adding title
        df.rename(columns=lambda x: x[4:] + '_' + title[15:22], inplace=True)
        if state:
            df_stats = stat_calc(df)
            df_all_stats = pd.concat([df_all_stats, df_stats], axis=0)
        if fft:
            df_fft = get_fft(df)
            #plot = df_fft.plot(title="FFT "+title, xlabel="Frequency (Hz)", ylabel="Amplitude", logy=True, xlim=(0,5000))        
            df_all_fft = pd.concat([df_all_fft, df_fft], axis=1)
        if psd:
            df_psd = get_psd(df)
            # df_psd.plot(title="PSD: power spectral density", xlabel="Frequency (Hz)", ylabel="Acceleration (g^2/Hz)", logy=True)
            df_all_psd = pd.concat([df_all_psd, df_psd], axis=1)
    workbook.close()
    if state:
        df_all_stats.to_excel(state_result_filename, sheet_name='state')
    if fft:
        df_all_fft.to_excel(fft_result_filename, sheet_name='fft')
    if psd:
        df_all_psd.to_excel(psd_result_filename, sheet_name='psd')

def compare_peak_from_fftdataframe(df: pd.DataFrame):
    peak_set = set()
    for col_name in df.columns:
        series = df[col_name].to_numpy()
        peak_idxs = annotate_peak(a=series, freq=df.index, prominence=0.25*series)
        peak_set.update(peak_idxs)
    print("peak numbers: %d"%len(peak_set))
    return peak_set

if __name__ == '__main__':
    acc_file_h = "d:\\cindy_hsieh\\My Documents\\project\\vibration_analysis\\test_data\\raw_data_20240308\\richard\\20240222Level_vs_Time.xlsx"
    acc_file_v = "d:\\cindy_hsieh\\My Documents\\project\\vibration_analysis\\test_data\\raw_data_20240308\\richard\\20240201Level_vs_Time.xlsx"
    fft_file_h = "d:\\cindy_hsieh\\My Documents\\project\\vibration_analysis\\test_data\\fft(horizontal)_202402.xls"
    fft_file_v = "d:\\cindy_hsieh\\My Documents\\project\\vibration_analysis\\test_data\\fft(vertical)_202402.xls"
    #acc_processing(acc_file_v, state=True, fft=True)
    #df_fft = pd.read_excel('fft.xlsx', index_col=0, header=0)
    df_fft = pd.read_excel(fft_file_h, skiprows=13, usecols="A:D", index_col=0)
    peak_set = compare_peak_from_fftdataframe(df_fft)
    for peak in peak_set:
        freq = df_fft.index[peak]
        if freq < 5000:
            save_bar_plot(df_fft.columns, df_fft.iloc[peak], '%d Hz FFT peak Amplitude'%freq, '%d.png'%freq)
    #plt.show()
    
#imf_x = imf[:,0,:] #imfs corresponding to 1st component
#imf_y = imf[:,1,:] #imfs corresponding to 2nd component
#imf_z = imf[:,2,:] #imfs corresponding to 3rd component
#axes_x = emd.plotting.plot_imfs(imf_x.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#axes_y = emd.plotting.plot_imfs(imf_y.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#axes_z = emd.plotting.plot_imfs(imf_z.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#plt.close("all")
