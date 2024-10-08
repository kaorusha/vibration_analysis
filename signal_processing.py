from math import comb
from operator import index
from typing import Any, List
from typing import Literal
import librosa
import matplotlib.axes
import matplotlib.mlab
from pyparsing import alphas
from scipy import signal
import numpy as np
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import emd
import memd.MEMD_all
import os

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
        plt.show()
    return output

def fft(df:pd.DataFrame, fs = 1, nperseq=8192, noverlap=8192//2, axis=1,
        domain:Literal["frequency", "order"] = "frequency", fg_column=3, pulse_per_round=2,
        rps:list = None, cols = 3):
    """
    do FFT with hanning window frame
    
    :param df: input acc data, each column represents a sequence signal of accelerometers, and the last column is the fg sensor.
    :param fs: sampling frequency
    :param nperseq: number of samples of each window frame
    :param noverlap: overlapping samples
    :param axis: rfft along windowed frame data axis, the shape of windowed frame is [number of frames, nperseq, columns number of input df]
    :param domain: units of the spectrum x labels
            'frequency': hz
            'order': relative to the inner race rotation, fr.
    :param fg_column: the last column number of input data frame
    :param pulse_per_round: fg sensor pulse numbers per round
    :param rps: round per second of each frame as a list with same length of frames 
    """
    frames = librosa.util.frame(df, frame_length=nperseq, hop_length=int(nperseq-noverlap), axis=0)
    window = np.hanning(nperseq)
    windowed_frames = np.empty(frames.shape) # frame.shape = [frame_numbers, frame_length, columns_dataframe]
    for col in range(frames.shape[-1]):
        np.multiply(window, frames[:,:,col], out=windowed_frames[:,:,col])
    sp = np.fft.rfft(windowed_frames, n=nperseq, axis=axis, norm='backward')
    freq = np.fft.rfftfreq(n=nperseq, d=1./fs)
    if domain == 'frequency':
        rms_averaged = np.sqrt(np.mean(np.power(np.abs(sp),2), axis=0))
        sp_rms = pd.DataFrame(data=rms_averaged, columns=df.columns)
        sp_rms['Frequency (Hz)'] = freq
        sp_rms.set_index('Frequency (Hz)', inplace=True)
        return sp_rms
    if domain == 'order':
        # the rotating freq of each frame
        if rps == None:
            idx = np.argmax(np.abs(sp[:, :, fg_column]), axis=1)
            rps = freq[idx]/pulse_per_round
        sp_dict = {}
        for i in range(len(rps)):
            keys = freq/rps[i]
            for j in range(len(keys)):
                if keys[j] in sp_dict:
                    sp_dict[keys[j]] = np.vstack([sp_dict[keys[j]],sp[i,j, :cols]])
                else:
                    sp_dict.update({keys[j]: np.array(sp[i,j, :cols])})
        print('There are %d order number as indexing'%len(sp_dict.keys()))
        sp_rms = pd.DataFrame(columns=df.columns[:cols])
        for key in sp_dict.keys():
            rms_averaged = np.sqrt(np.mean(np.power(np.abs(sp_dict[key]),2), axis=0))
            sp_rms.loc[key] = rms_averaged
        sp_rms.sort_index(inplace=True)
        sp_rms.index.rename('order of rotating frequency', inplace=True)
        return sp_rms

def get_fft(df: pd.DataFrame, cut_off_freq = 0, fs = 48000, frame_len=8192, overlap = 0.75,
            domain: Literal["frequency", "order"] = "frequency", fg_column=3, pulse_per_round = 2, rps = None, cols = None):
    '''
    return fft as dataframe type
    
    :param df: input level vs time signal
    :param cut_off_freq: cut off frequency for high-pass filter

    :param fs: sampling frequency
    :param domain: units of the spectrum x labels
                'frequency': hz
                'order': relative to the inner race rotation, fr.
    signal processing step:
    1. subtract dc bias from accelerometer data
    2. high pass filter (optional)
    3. do FFT with hanning window frame
    4. get mean of FFT spectrum
    '''
    # subtract dc bias from acc data
    detrend_df = df - np.mean(df.to_numpy(), axis=0)
    if cut_off_freq > 0:
        # use high-pass filter to remove dc 
        filtered_df = butter_highpass(detrend_df, df.index, cut_off_freq, fs, 2, visualize=True)
        filtered_df = pd.DataFrame(data=filtered_df, columns=df.columns)
        df_fft = fft(filtered_df, fs=fs, nperseq=frame_len, noverlap=frame_len*overlap, domain=domain, fg_column = fg_column, pulse_per_round=pulse_per_round, rps=rps, cols = cols)
    else:
        df_fft = fft(detrend_df, fs=fs, nperseq=frame_len, noverlap=frame_len*overlap, domain=domain, fg_column=fg_column, pulse_per_round=pulse_per_round, rps=rps, cols = cols)
    return df_fft

def annotatePeaks(a: Any, freq: Any, ax: matplotlib.axes.Axes = None, prominence:Any|None = None):    
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

def acc_processing(hdf_level_time_filename:str,
                   rename_column_method = None,
                   usecols:list = None,
                   cols:int = 3,
                   sheets:list = None,
                   state: bool = False, state_result_filename:str = 'state.xlsx', 
                   fft: bool = False, fft_result_filename:str = 'fft.xlsx',
                   cut_off_freq: float = 60,
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
    if sheets == None:
        sheets = workbook.sheetnames
    for sheet in sheets:
        title = workbook[sheet]["B5"].value
        df = pd.read_excel(hdf_level_time_filename, sheet_name=sheet, header=0, index_col=0, skiprows=13, usecols=usecols)
        # rewrite column title adding title
        if rename_column_method is not None:
            rename_column_method(df, title)
        if state:
            df_stats = stat_calc(df)
            df_all_stats = pd.concat([df_all_stats, df_stats], axis=0)
        if fft:
            df_fft = get_fft(df, cut_off_freq=cut_off_freq, cols=cols)
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
    peak_dict = {}
    for col_name in df.columns:
        series = df[col_name].to_numpy()
        peak_idxs = annotatePeaks(a=series, freq=df.index, prominence=0.25*series)
        update_peak_dic(peak_dict, peak_idxs)
    print("peak numbers: %d"%len(peak_dict.keys()))
    return peak_dict

def update_peak_dic(dic:dict, idxs:list[int]):
    for key in idxs:
        if key in dic.keys():
            dic[key] += 1
        else:
            dic[key] = 1

def rename_col(df: pd.DataFrame, title:str):
    # rewrite column title adding title
    df.rename(columns=lambda x: title[15:22] + '_' + x.split()[0][4:], inplace=True)

def fft_processing(fft_filename:str, file_type:Literal['hdf', 'normal'] = 'normal', rename_column_method = None, usecols = None, combine = True):
    """
    read previous exported FFT excel file, loop for each sheet, combine as one pandas data frame, or return a dictionary of dataframe
    """
    workbook = openpyxl.load_workbook(fft_filename, read_only=True, data_only=True, keep_links=False)
    print("There are %d"%len(workbook.sheetnames) + " sheets in this workbook ( " + fft_filename + " )")

    if file_type == 'normal':
        df_dict = pd.read_excel(fft_filename, sheet_name=None, header=0, index_col=0, usecols=usecols)
    if file_type == 'hdf':
        df_dict = pd.read_excel(fft_filename, sheet_name=None, header=0, index_col=0, skiprows=13, usecols=usecols)
        for sheet in workbook.sheetnames:
            title = workbook[sheet]["B5"].value
            rename_column_method(df_dict[sheet], title)
    if combine == False:
        return df_dict 
    # combine all fft to the same dataframe
    df_all_fft = pd.DataFrame()
    for sheet in workbook.sheetnames:
        df_all_fft = pd.concat([df_all_fft, df_dict[sheet]], axis=1)
    workbook.close()
    return df_all_fft

def class_average_peak(peak_dic:dict, df_fft: pd.DataFrame):
    idx_list = []
    for peak in peak_dic:
        freq = df_fft.index[peak]
        if freq > 5000:
            continue
        if peak_dic.get(peak) < 2:
            continue
        idx_list.append(peak)

    return df_fft.iloc[idx_list].mean()

def acc_processing_ver2(dir:str, 
                        state: bool = False, state_result_filename:str = 'state.xlsx', cols = None,
                        fft: bool = False, fft_result_filename:str = 'fft.xlsx', domain: Literal["frequency", "order"] = "frequency"):
    """
    read level vs time .xlsx file, loop for each file in the directory, read as panda data frame and do selected processing, 
    save the result of from multiple file of raw acc data into seperate excel sheet. Because order is representes as number
    of times of rotation frequency, the orders of each acc file will be different since the rotation frequency is changing.
    The FFT result should save in different sheet as the indexing order is different.
    
    :param state: whether use the data frame to calculate time domain standard deviation etc..
    :param fft: whether to calculate fast fourier transform
    :param domain: units of the spectrum x labels
            'frequency': hz
            'order': relative to the inner race rotation, fr.
    """
    if state:
        df_all_stats = pd.DataFrame()
    if fft:
        # an excel file with one default sheet is created
        wb = openpyxl.Workbook()
        wb.save(fft_result_filename)
        wb.close()

    for file_name in os.listdir(dir):
        if file_name.endswith('.xlsx'):
            df = pd.read_excel(dir+file_name, header=0)
            print("read excel %s"%file_name)
            #df.rename(columns=lambda x: file_name[0:9] + '_' + x, inplace=True)
            if state:
                df_stats = stat_calc(df)
                df_all_stats = pd.concat([df_all_stats, df_stats], axis=0)
            if fft:
                df_fft = get_fft(df, fs=51200, domain=domain, cols=cols)
                with pd.ExcelWriter(fft_result_filename, mode="a", if_sheet_exists="new", engine="openpyxl") as writer:
                    df_fft.to_excel(writer, sheet_name=file_name[:-5])

    if state:
        df_all_stats.to_excel(state_result_filename, sheet_name='state')
    if fft:
        # remove the first default blank sheet
        with pd.ExcelWriter(fft_result_filename, mode="a", if_sheet_exists="new", engine="openpyxl") as writer:
            writer.book.remove(writer.book['Sheet'])
            writer.book.save(fft_result_filename)
            writer.book.close()

def savefftplot(df_fft:pd.DataFrame, sample:list, annotate_peaks:bool, annotate_bends:bool, save_fig:bool, save_dir:str):
    '''
    show or save a fft plot of selected sample numbers

    :param df_fft: each column represents a accelerometer fft spectrum, first six number is the part number, and there are 8 column for each part,
    :param which is left/right/axile/fg/up/down/axile/fg, fg signal will not be shown in the figure, so the cols = [0,1,2,4,5,6]
    :param sample: the selected sample number to show or save the picture, array of integer
    :param annotate_peaks: True to annotate peaks of fft spectrum
    :param annotate_bends: True to annotate frequency bends
    :param save_fig: True to save the fig 
    :param save_dir: save the figure to this directory
    '''
    for n in sample:
        series = df_fft[df_fft.columns[8*n]].to_numpy()
        cols = [8*n,8*n+1,8*n+2,8*n+4,8*n+5,8*n+6]
        ax = df_fft.iloc[:, cols].plot(title="FFT ", xlabel="Frequency (Hz)", ylabel="Amplitude", logy=True, xlim=(0,5000), layout="constrained",figsize=(10,10))
        if annotate_peaks:
            peak_idxs = annotatePeaks(a=series, freq=df_fft.index, ax=ax, prominence=0.25*series)
        if annotate_bends:
            fb0 = bearingFaultBands(rotationSpeed(df_fft, n), 6, 1.5875, 5.645, 0, width=0.1)
            annotateFreqBands(ax, fb0, df_fft.index)
        if save_fig:
            plt.savefig(save_dir+df_fft.columns[8*n][:6], transparent=False, dpi=80, bbox_inches="tight")
        plt.show()

class BearingFaultBands:
    class Info:
        def __init__(self):
            self.Centers = []
            self.Labels = []
            self.FaultGroups = []
        def __str__(self):
            return "Centers: %s,\nLabels: %s,\nFaultGroups: %s"%(self.Centers, self.Labels, self.FaultGroups)
    def __init__(self):
        self.fault_bands = np.ndarray([], dtype=float)
        self.info = BearingFaultBands.Info()
    
    def __str__(self):
        print('fault bands = \n%s'%self.fault_bands)
        print('info = struct with fields: \n%s'%self.info)
        return ''
        
    def insertDict(self, info_insert:list):
        self.info.Centers.append(info_insert[0])
        self.info.Labels.append(info_insert[1])
        self.info.FaultGroups.append(info_insert[2])
    
    def countWidth(self, width: float):
        fault_bands_list = []
        for i in self.info.Centers:
            fault_bands_list.append([i - width, i + width])
        self.fault_bands = np.array(fault_bands_list)
    
    def countDomain(self,fr:float, domain:Literal["frequency", "order"] = "frequency"):
        self.info.Centers = np.array(self.info.Centers)
        if domain == "frequency":
            self.info.Centers *= fr
            self.fault_bands *= fr

def bearingFaultBands(fr:float, nb:int, db:float, dp:float, beta:float, harmonics = [1], sidebands = [0], width:float = 0.1, domain:Literal["frequency", "order"] = "frequency"):
    '''
    https://www.mathworks.com/help/predmaint/ref/bearingfaultbands.html
    the calculation is based on fixed outer race with rotating inner race
    
    :param fr: Rotational speed of the shaft or inner race, this parameter is used if the domain is 'frequency'.
    :param nb: Number of balls or rollers
    :param db: Diameter of the ball or roller
    :param dp: Pitch diameter
    :param beta: Contact angle in degree
    :param harmonics: harmonics of the fundamental frequency to be included
    1 (default) | vector of positive integers
    :param Sidebands: Sidebands around the fundamental frequency and its harmonics to be included
    0 (default) | vector of nonnegative integers
    :param width: width of the frequency bands centered at the nominal fault frequencies
    :param domain: units of the fault band frequencies
            'frequency': hz
            'order': relative to the inner race rotation, fr.
    
    output
    ------
    ### fb - Fault frequency bands, returned as an N-by-2 array, where N is the number of fault frequencies. 
    FB is returned in the same units as FR, in either hertz or orders depending on the value of 'Domain'. 
    Use the generated fault frequency bands to extract spectral metrics using faultBandMetrics. 
    The generated fault bands are centered at:
    * Outer race defect frequency, Fo, and its harmonics
    * Inner race defect frequency, Fi, its harmonics and sidebands at FR
    * Rolling element (ball) defect frequency, Fbits harmonics and sidebands at Fc
    * Cage (train) defect frequency, Fc and its harmonics
    The value W is the width of the frequency bands, which you can specify using the 'Width' name-value pair.
    ### Info - Information about the fault frequency bands in FB, returned as a structure with the following fields:
    * Centers — Center fault frequencies
    * Labels — Labels describing each frequency
    * FaultGroups — Fault group numbers identifying related fault frequencies
    '''
    alpha = np.cos(beta * np.pi / 180)
    Fc_order = 0.5 * (1 - db/dp * alpha)
    Fb_order = 0.5 * (dp/db - db/dp * alpha**2)
    Fo_order = nb * Fc_order
    Fi_order = nb * (1 - Fc_order)
    res = BearingFaultBands()
    for i in harmonics:
        res.insertDict([Fo_order * i, '%dFo'%i, 1])
        for j in sidebands:
            fi = Fi_order * i
            fb = Fb_order * i
            if j > 0:
                res.insertDict([fi - j,             '%dFi-%dFr'%(i,j), 2])
            res.insertDict([    fi,                 '%dFi'%i,          2])
            if j > 0:
                res.insertDict([fi + j,             '%dFi+%dFr'%(i,j), 2])
                res.insertDict([fb - j * Fc_order,  '%dFb-%dFc'%(i,j), 3])
            res.insertDict([    fb,                 '%dFb'%i,          3])
            if j > 0:
                res.insertDict([fb + j * Fc_order,  '%dFb+%dFc'%(i,j), 3])
        res.insertDict([Fc_order * i, '%dFc'%i, 4])
    res.countWidth(width)
    res.countDomain(fr, domain)
    return res

def rotationSpeed(df_fft: pd.DataFrame, sample_no:int):
    speed_lr = df_fft.iloc[:,sample_no * 8 + 3].idxmax() * 0.5
    speed_ud = df_fft.iloc[:,sample_no * 8 + 7].idxmax() * 0.5
    return (speed_lr + speed_ud) / 2

def binary_search(arr, low, high, x):
    '''
    given an array, and its lowest index and highest index, return index closest to the value x
    '''
    if high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        return low

def annotateFreqBands(axes: matplotlib.axes.Axes, fb: BearingFaultBands, alpha):
    '''
    Annotate bearing frequency bands in different color based on its fault group.
    Note: the input axes should use constrained layout
    
    :param axes: subplots for annotation
    :param fb: bearing fault bands
    :param x: x-axis of subplots, which is an array with a specified range and step increment
    '''
    left, right = axes.get_xlim()
    step = (fb.fault_bands[0][1] - fb.fault_bands[0][0])/2
    x = np.arange(left, right, step)
    length = len(x)
    color_arr = ['red', 'orange', 'green', 'blue']
    mask_arr = []
    for i in range(0,4):
        mask_arr.append(np.zeros(length))
    
    for (x1, x2), g, s in zip(fb.fault_bands, fb.info.FaultGroups, fb.info.Labels):
        if (x1 > x[-1]): 
            continue
        mask_arr[g - 1][binary_search(x, 0, length - 1, x1):binary_search(x, 0, length - 1, x2)] = 1
        axes.annotate(s, xy=(x1, 0.95), xycoords=('data', 'subfigure fraction'), rotation='vertical', verticalalignment='top')
    
    for i in range(0, 4):
        axes.fill_between(x, 0, 1, where= mask_arr[i], color= color_arr[i], alpha=alpha, transform=axes.get_xaxis_transform())
    
    axes.annotate('Cage defect frequency, Fc\nBall defect frequency, Fb\nOuter race defect frequency, Fo\nInner race defect frequency, Fi\n',
                  xy=(0.95, 0.05), xycoords='subfigure fraction', horizontalalignment='right')
    
def techometer(fg_signal: pd.DataFrame, thereshold:float, fs:int, pulse_per_round: int):
    '''
    :param fg_signal: square wave signal array
    :param thereshold: count for rising edge
    :param fs: smapling frequency
    :param pulse_per_round: pulse numbers per round, used for rotation speed calculation
    
    returns
    ------
    rotating speed rps= round per second (hz)
    '''
    state = False
    delta = 0
    rps = pd.Series(np.zeros(len(fg_signal)))
    time_buffer = fs/200
    for i in range(1, len(fg_signal), 1):
        delta += 1
        if fg_signal[i] > thereshold and state == False:
            # preventing zero as denominator
            if (i < time_buffer):
                continue
            rps[i] = fs / (delta * pulse_per_round)
            delta = 0
            state = True
        else:
            rps[i] = rps[i - 1]
            if fg_signal[i] < thereshold:
                state = False
    return rps

def fg_fft(fg_signal: pd.DataFrame, fs = 1, nperseq=8192, noverlap=8192//2):
    '''
    use fg signal to get the fft and find the rotation speed in hz
    '''
    detrend_df = fg_signal - np.mean(fg_signal.to_numpy(), axis=0)
    frames = librosa.util.frame(detrend_df, frame_length=nperseq, hop_length=int(nperseq-noverlap))
    window = np.hanning(nperseq)
    windowed_frames = np.empty(frames.shape)
    for col in range(frames.shape[-1]):
        np.multiply(window, frames[:,col], out=windowed_frames[:,col])
    sp = np.fft.rfft(windowed_frames, n=nperseq, norm='backward', axis=0)
    freq = np.fft.rfftfreq(n=nperseq, d=1./fs)
    rps = pd.Series(np.zeros(len(fg_signal)))
    idx = np.argmax(np.abs(sp), axis=0)
    hop_lenth = int(nperseq - noverlap)
    start = 0
    for i in idx:
        rps[start:start + hop_lenth] = freq[i]/2
        start += hop_lenth
    # ending samples
    rps[start:] = rps[start - 1]    
    return rps

def level_and_rpm_seperate_processing(hdf_level_time_filename:str, level_sheet:str, level_col:list, fs = 48000, 
                                      hdf_rpm_time_filename = None, rpm_sheet = None, fs_rpm = 1000,
                                      nperseq = 8192, overlap = 0.75, fft_filename = None, fft_sheet = None):
    '''
    read seperate **level_vs_time.hdf** and **rpm_vs_time.hdf**, with different sampling frequency, to calculate
    FFT based on rpm rotating speed, in order to compare different sample with normalized rotating frequency order
    and output the FFT order result to specified file

    :param hdf_level_time_filename: level_vs_time.hdf transported excel
    :param level_sheet: sheet name
    :param level_col: used columns
    :param fs: sampling frequency
    :param hdf_rpm_time_filename: rpm_vs_time.hdf transported excel
    :param rpm_sheet: sheet name
    :param fs_rpm: rpm_vs_time.hdf sampling frequency, if it is lower than lever_vs_time, than use duplicate for sample augmentation
    :param nperseq: number of sample per frame
    :param overlap: percentage of overlape
    :param fft_filename: output fft file name
    :param fft_sheet: output fft sheet name, can be append to an exist fft result file as seperate sheet 

    example:
    sound_hdf = '../../test_data//20240808//good-100%-18300.Level vs. Time.xlsx'
    rpm_hdf = '../../test_data//20240814//1833-20%.RPM vs. Time.xlsx'
    fft_file = '../../test_data//20240808//fft_order.xlsx'
    level_and_rpm_seperate_processing(hdf_level_time_filename=sound_hdf, hdf_rpm_time_filename=rpm_hdf, level_sheet='Sheet22', level_col=[0,1], rpm_sheet='Sheet1',
                                      fft_filename=fft_file, fft_sheet='1833')
    '''
    workbook = openpyxl.load_workbook(hdf_level_time_filename, read_only=True, data_only=True, keep_links=False)
    title = workbook[level_sheet]["B5"].value
    df = pd.read_excel(hdf_level_time_filename, sheet_name=level_sheet, header=0, index_col=0, skiprows=13, usecols=level_col)
    # rewrite column title adding title
    df.rename(columns=lambda x:title.split()[0], inplace=True)
    df_rpm = pd.read_excel(hdf_rpm_time_filename, sheet_name=rpm_sheet, header=0, index_col=0, skiprows=13, usecols="A:B")
    # calculate frame
    df_rpm = df_rpm.loc[df_rpm.index.repeat(fs/fs_rpm)]
    df_rpm.reset_index(drop=True, inplace=True)
    frames = librosa.util.frame(df_rpm, frame_length=nperseq, hop_length=int(nperseq*(1-overlap)), axis=0)
    rps = []
    for frame in frames:
        rps.append(np.round(np.mean(frame, axis=0)/60))
    df_fft = get_fft(df, cut_off_freq = 10, fs = fs, frame_len=nperseq, overlap = overlap,
                     domain="order", pulse_per_round = 2, rps = rps, cols = 1)
    # if file not exist
    if not os.path.exists(fft_filename):
        wb = openpyxl.Workbook()
        wb.save(fft_filename)
        wb.close()
    with pd.ExcelWriter(fft_filename, mode="a", if_sheet_exists="new", engine="openpyxl") as writer:
        df_fft.to_excel(writer, sheet_name=fft_sheet)
        if 'Sheet' in writer.book.sheetnames:
            writer.book.remove(writer.book['Sheet'])
        writer.book.save(fft_filename)
        writer.book.close()
    
def compare_rps_of_rpm_vs_time_file(dir):
    df_dict = {}

    for file_name in os.listdir(dir):
        if file_name.endswith('.xlsx') and not file_name.startswith('~$'):
            print('opening file: %s'%file_name)
            wb = openpyxl.load_workbook(dir + file_name, read_only=True, data_only=True, keep_links=False)
            title = wb['Sheet1']["B5"].value
            wb.close()
            df = pd.read_excel(dir + file_name, sheet_name='Sheet1', header=0, index_col=0, skiprows=13)
            df[df.columns[0]] = df[df.columns[0]]/60
            key = title.split()[0]
            df.rename(columns=lambda x: title.split()[0], inplace=True)
            df_dict[key] = df
    return df_dict

def acc_processing_coherence(good_sample_dir:str, dir: str, good_sample_num:str, result_filename:str, visualize:bool = False):
    wb = openpyxl.Workbook()
    wb.save(result_filename)
    wb.close()
    df_x_lr = pd.read_excel(good_sample_dir + good_sample_num + '_lr.xlsx', header = 0)
    df_x_ud = pd.read_excel(good_sample_dir + good_sample_num + '_ud.xlsx', header = 0)
    
    for file_name in os.listdir(dir):
        if file_name.endswith('.xlsx') and not file_name.startswith(good_sample_num):
            print("read excel %s"%file_name)
            df_y = pd.read_excel(dir + file_name, header = 0)
            if visualize:
                fig, axs = plt.subplots(4, 1, layout='constrained')
            if 'lr' in file_name:
                freq, Cxy = signal.coherence(df_x_lr, df_y, fs=51200, nperseg=8192, noverlap=8192*0.75, axis=0)
                if visualize:
                    for i in [0,1,2,3]:
                        cxy, f = axs[i].cohere(x=df_x_lr.iloc[:, i], y=df_y.iloc[:, i], NFFT=8192, Fs=51200, detrend='mean', noverlap=int(8192*0.75), window=np.hanning(8192),
                                            label=df_x_lr.columns[i] + ' vs ' + df_y.columns[i])
                        axs[i].legend()

            if 'ud' in file_name and visualize:
                freq, Cxy = signal.coherence(df_x_ud, df_y, fs=51200, nperseg=8192, noverlap=8192*0.75, axis=0)
                if visualize:
                    for i in [0,1,2,3]:
                        cxy, f = axs[i].cohere(x=df_x_ud.iloc[:, i], y=df_y.iloc[:, i], NFFT=8192, Fs=51200, detrend='mean', noverlap=int(8192*0.75), window=np.hanning(8192),
                                            label=df_x_ud.columns[i] + ' vs ' + df_y.columns[i])
                        axs[i].legend()
            if visualize:
                plt.show()
            
            with pd.ExcelWriter(result_filename, mode="a", if_sheet_exists="new", engine="openpyxl") as writer:
                df = pd.DataFrame(data=Cxy, index=freq, columns=df_y.columns)
                df.to_excel(writer, sheet_name=file_name[:-5])
    # remove the first default blank sheet
    with pd.ExcelWriter(result_filename, mode="a", if_sheet_exists="new", engine="openpyxl") as writer:
        writer.book.remove(writer.book['Sheet'])
        writer.book.save(result_filename)
        writer.book.close()

def corr(df:pd.DataFrame, result_filename:str):
    wb = openpyxl.Workbook()
    wb.save(result_filename)
    wb.close()
    
    for meth in ['pearson', 'kendall', 'spearman']:
        df_corr = df.corr(method=meth)
        with pd.ExcelWriter(result_filename, mode='a', engine='openpyxl') as writer:
            df_corr.to_excel(writer, sheet_name=meth)
    # remove the first default blank sheet
    with pd.ExcelWriter(result_filename, mode="a", if_sheet_exists="new", engine="openpyxl") as writer:
        writer.book.remove(writer.book['Sheet'])
        writer.book.save(result_filename)
        writer.book.close()

def fft_analysis(good_sample_fft: {pd.DataFrame}, benchmarks_sheet:str, abnormal_sample_fft: {pd.DataFrame}, types: list):
    '''
    if the amplitude of the index frequency of all abnormal samples is greater than all good samples, 
    high light that frequency on the output spectrum plot.
    '''
    fig, axs = plt.subplots(len(types), 1, layout='constrained')
    
    bool_df = pd.DataFrame(columns=types)
    for freq in good_sample_fft[benchmarks_sheet].index:
        # find the max of each type
        maximum = [0 for _ in range(len(types))]
        for sheet in good_sample_fft:
            idx = binary_search(good_sample_fft[sheet].index, 0, len(good_sample_fft[sheet].index) - 1, freq)
            for col in range(len(good_sample_fft[sheet].columns)):
                for i in range(len(types)):
                    if good_sample_fft[sheet].columns[col].endswith(types[i]):
                        maximum[i] = max(good_sample_fft[sheet].iloc[idx, col], maximum[i])
                        break
        # check each type is larger than the good samples or not
        bools = pd.DataFrame(data=[[True for i in types]], columns=types)

        for sheet in abnormal_sample_fft:
            idx = binary_search(abnormal_sample_fft[sheet].index, 0, len(abnormal_sample_fft[sheet].index) - 1, freq)
            for col in range(len(abnormal_sample_fft[sheet].columns)):
                for i in range(len(types)):
                    if abnormal_sample_fft[sheet].columns[col].endswith(types[i]):
                        bools.iloc[:, i] = ((abnormal_sample_fft[sheet].iloc[idx, col] > maximum[i]) & bools.iloc[:, i])
                        break
        bool_df = pd.concat([bool_df, bools], axis=0)
    # high light the frequency
    for i in range(len(types)):
        axs[i].fill_between(good_sample_fft[benchmarks_sheet].index, 0, 1, where= bool_df.iloc[:,i], color= 'red', alpha=0.5, transform=axs[i].get_xaxis_transform())
        axs[i].set_xlim(left=good_sample_fft[benchmarks_sheet].index[0], right=7600/44)
        axs[i].set_yscale('log')
        axs[i].set_ylabel(types[i])
    
    # add spectrum of good and bad samples
    plot_df_each_col_a_fig(abnormal_sample_fft, types, axs, color='orange', linewidth=1, alpha=0.5)
    plot_df_each_col_a_fig(good_sample_fft, types, axs, color='green', linewidth=1, alpha=0.5)
    plt.show()

def plot_df_each_col_a_fig(df_dict:{pd.DataFrame}, types: list, axs: np.ndarray, **arg):
    for sheet in df_dict:
        for col_name in df_dict[sheet].columns:
            for i in range(len(types)):
                if col_name.endswith(types[i]):
                    axs[i].plot(df_dict[sheet].index, df_dict[sheet][col_name], **arg)
                    break

if __name__ == '__main__':
    #savefftplot(df_fft, [0], False, True, False, acc_file_dir)
    #peak_dict = compare_peak_from_fftdataframe(df_fft)
    #print(class_average_peak(peak_dict, df_fft))
    #df_fft = pd.read_excel('sp_rms.xlsx', header=0, index_col=0)
    #fig, ax = plt.subplots(layout='constrained')
    #df_fft.plot(logx=True, title='ordered frequency spectrum', xlabel='order of rotation speed', ylabel='Amplitude', logy=True, ax=ax)
    '''
    fb = bearingFaultBands(fr=44, nb=6, db=1.5875, dp=5.645, beta=0, harmonics=range(1, 40), domain='order', width=0.1)
    for sheet in df:
        if not (sheet.startswith('001833') or sheet.startswith('004073')):
            continue
        for col in range(len(df[sheet].columns)):
            fig, ax = plt.subplots(layout='constrained')
            ax.plot(df[sheet].index, df[sheet].iloc[:, col], color=color['blue'], label=df[sheet].columns[col])
            if sheet.startswith('001833'):
                ax.plot(df_sound.index, df_sound['1833-20%'], color=color['red'], label='sound')
            if sheet.startswith('004073'):
                ax.plot(df_sound.index, df_sound['4073-20%-2680'], color=color['red'], label='sound')
            if sheet.endswith('lr'):
                ax.plot(df['000045_lr'].index, df['000045_lr'].iloc[:, col], color=color['green'], label=df['000045_lr'].columns[col])
            if sheet.endswith('ud'):
                ax.plot(df['000045_ud'].index, df['000045_ud'].iloc[:, col], color=color['green'], label=df['000045_ud'].columns[col])
            ax.set_xlim(0, 115)
            ax.set_yscale("log")
            ax.set_xlabel('order of rotating frequency')
            ax.set_ylabel('Amplitude')
            ax.legend()
            annotateFreqBands(ax, fb, 0.4)     
            plt.show()
            break
    '''
#imf_x = imf[:,0,:] #imfs corresponding to 1st component
#imf_y = imf[:,1,:] #imfs corresponding to 2nd component
#imf_z = imf[:,2,:] #imfs corresponding to 3rd component
#axes_x = emd.plotting.plot_imfs(imf_x.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#axes_y = emd.plotting.plot_imfs(imf_y.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#axes_z = emd.plotting.plot_imfs(imf_z.transpose(), time_vect=df.index.to_numpy(), sharey=False)
#plt.close("all")
