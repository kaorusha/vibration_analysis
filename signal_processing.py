from turtle import distance
from typing import Any, List
from typing import Literal
import librosa
import matplotlib.axes
from scipy import signal
import numpy as np
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial
import memd.MEMD_all
import os
from gwpy.timeseries import TimeSeries

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

def fft(df:pd.DataFrame, fs = 1, nperseq=8192, window = np.hanning(8192), noverlap=8192//2, axis=1, fg_column=3, pulse_per_round=2, rps:list = None):
    """
    do FFT with window frame, return complex number of each window frame, and if the rps is None also fg_column is provided, return the rps of each frame
    
    :param df: input acc data, each column represents a sequence signal of accelerometers, and the last column is the fg sensor.
    :param fs: sampling frequency
    :param nperseq: number of samples of each window frame
    :param noverlap: overlapping samples
    :param axis: rfft along windowed frame data axis, the shape of windowed frame is [number of frames, nperseq, columns number of input df]
    :param fg_column: fg signal as square wave, usually the last column number of input data frame, unused if rps is given.
    :param pulse_per_round: fg sensor pulse numbers per round
    :param rps: rounds per second of each frame as a list with same length of frames, if not provided than should give fg_column
    """
    frames = librosa.util.frame(df, frame_length=nperseq, hop_length=int(nperseq-noverlap), axis=0)
    windowed_frames = np.empty(frames.shape) # frame.shape = [frame_numbers, frame_length, columns_dataframe]
    for col in range(frames.shape[-1]):
        np.multiply(window, frames[:,:,col], out=windowed_frames[:,:,col])
    sp = np.fft.rfft(windowed_frames, n=nperseq, axis=axis, norm='backward')
    freq = np.fft.rfftfreq(n=nperseq, d=1./fs)
    # get the rotating freq of each frame
    if rps == None:
        idx = np.argmax(np.abs(sp[:, :, fg_column]), axis=1)
        rps = freq[idx]/pulse_per_round
    
    if len(rps) != sp.shape[0]:
        raise ValueError('rps is the rotating frequency of each frame, should be the same length of frames')    

    return freq, sp, rps
    
def get_fft(df: pd.DataFrame, cut_off_freq = 0, fs = 48000, frame_len=8192, noverlap = 8192*0.75,
            domain: Literal["frequency", "order"] = "frequency", fg_column=3, pulse_per_round = 2, rps = None, cols = None,
            average: Literal["rms", "mean", "median"] = "rms"):
    '''
    return fft as dataframe type
    
    :param df: input level vs time signal
    :param cut_off_freq: cut off frequency for high-pass filter

    :param fs: sampling frequency
    :param domain: units of the spectrum x labels
                'frequency': hz
                'order': relative to the inner race rotation, fr.
    :param fg_column: fg signal as square wave, usually the last column number of input data frame, unused if rps is given.
    :param pulse_per_round: fg sensor pulse numbers per round
    :param rps: rounds per second of each frame as a list with same length of frames, if not provided than should give fg_column.
                This parameter is used when the domain is 'order'.
    :param cols: use first int(cols) for fft analysis, this values usually equal to the number of accelerometers.
    :param average: the averaging method of windowed frames
    
    signal processing step:
    1. subtract dc bias from accelerometer data
    2. high pass filter (optional)
    3. do FFT with hanning window frame
    4. get average of FFT spectrum based on different return type of fft helper function
    '''
    # subtract dc bias from acc data
    detrend_df = df - np.mean(df.to_numpy(), axis=0)
    if cut_off_freq > 0:
        # use high-pass filter to remove dc
        detrend_df = butter_highpass(detrend_df, df.index, cut_off_freq, fs, 2, visualize=True)
    freq, sp, rps = fft(detrend_df, fs=fs, nperseq=frame_len, noverlap=noverlap, fg_column=fg_column, pulse_per_round=pulse_per_round, rps=rps)
    
    # Average over windows
    sp = sp[:,:,:cols] # get the used range
    if domain == 'frequency':
        if average == 'rms':
            sp_averaged = np.sqrt(np.mean(np.power(np.abs(sp),2), axis=0))
        elif average == 'median':
            # np.median must be passed real arrays for the desired result
            if np.iscomplexobj(sp):
                sp_averaged = (np.median(np.real(sp), axis=0) + 1j * np.median(np.imag(sp), axis=0))
            else:
                sp_averaged = np.median(sp, axis=0)
        elif average == 'mean':
            sp_averaged = sp.mean(axis=0)
        else:
            raise ValueError('choose from specified methods')
        sp_averaged = pd.DataFrame(data=sp_averaged, columns=df.columns[:cols])
        sp_averaged['Frequency (Hz)'] = freq
        sp_averaged.set_index('Frequency (Hz)', inplace=True)
    if domain == 'order':
        sp_dict = {}
        for i in range(len(rps)):
            keys = freq/rps[i]
            for j in range(len(keys)):
                if keys[j] in sp_dict:
                    sp_dict[keys[j]] = np.vstack([sp_dict[keys[j]],sp[i,j,:]])
                else:
                    sp_dict.update({keys[j]: np.array(sp[i,j,:])})
        print('There are %d order number as indexing'%len(sp_dict.keys()))
        # getting average
        sp_averaged = pd.DataFrame(columns=df.columns[:cols])
        for freq_order in sp_dict.keys():
            if average == 'rms':
                sp_order = np.sqrt(np.mean(np.power(np.abs(sp_dict[freq_order]),2), axis=0))
            elif average == 'median':
                if np.iscomplex(sp_dict[freq_order].all()):
                    sp_order = (np.median(np.real(sp_dict[freq_order]), axis=0) + 1j * np.median(np.imag(sp_dict[freq_order]), axis=0))
                else:
                    sp_order = np.median(sp_dict[freq_order], axis = 0)
            elif average == 'mean':
                sp_order = sp_dict[freq_order].mean(axis=0)
            else:
                raise ValueError('choose from specified methods')
            
            sp_averaged.loc[freq_order] = sp_order
        sp_averaged.sort_index(inplace=True)
        sp_averaged.index.rename('order of rotating frequency', inplace=True)
        
    return sp_averaged

def annotatePeaks(x: Any, y: Any, ax: matplotlib.axes.Axes = None, prominence:Any|None = None, dot = None, 
                  annotateX=True, annotateY=False,
                  rotation = 45, 
                  xytext=(0, 30), textcoords='offset pixels',
                  arrowprops = dict(facecolor='blue', arrowstyle="->", connectionstyle="arc3")):    
    """
    analysis the peak and add annotation on the graph

    :param x: 1d array index
    :param y: 1d array (note: if it is a spectrum, remember to use absolute value)
    
    """
    peaks, dic = signal.find_peaks(y, prominence=prominence)
    if ax != None:
        for idx in peaks:
            if dot != None:
                ax.plot(x[idx], y[idx], dot)
            string = ''
            if annotateX:
                string += '%.2f'%x[idx]
            if annotateY:
                string += ', %.2f'%y[idx]
            ax.annotate(string, 
                        xy=(x[idx], y[idx]), rotation=rotation, xycoords='data',
                        xytext=xytext, textcoords=textcoords,
                        arrowprops=arrowprops)
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
        #if x != arr[low]:
        #    print("binary search: find %f and get %f"%(x, arr[low]))
        return high

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
        if file_name.endswith('.xlsx'):
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

def csd_order(x:pd.DataFrame, y:pd.DataFrame, fs:int, nperseg:int, noverlap:int, cols:int, fg_column = 3,
              rps_x = None, rps_y = None, average:Literal['mean', 'median'] = 'mean'):
    win = np.hanning(nperseg)
    # detrend
    detrend_x = x - np.mean(x.to_numpy(), axis=0)
    freq_x, fft_x, rps_x = fft(detrend_x, fs=fs, nperseq=nperseg, window=win, noverlap=noverlap, fg_column=fg_column, rps=rps_x)
    if x.equals(y):
        freq_y, fft_y, rps_y = freq_x, fft_x, rps_x
    else:
        detrend_y = y - np.mean(y.to_numpy(), axis=0)
        freq_y, fft_y, rps_y = fft(detrend_y, fs=fs, nperseq=nperseg, window=win, noverlap=noverlap, fg_column=fg_column, rps=rps_y)
    
    Pxy = np.empty(fft_x[:,:,:cols].shape, dtype=np.complex128)
    # multiply, because the base frequency might be different, hence the frequency array is diveded by different value,
    # the resulting index is not the same for each frame.
    # a dictionary is used to store the multiply result at each order.
    Pxy_dict = {}
    
    # scaling for power spectral density
    scale = 1.0 / (fs * (win*win).sum())
    # input signal is real so the rfft amplitude *2, and if nperseq can not divided by 2, the last point is unpaired Nyquist freq point, don't double
    not_divided_by_2 = nperseg % 2

    for frame in range(fft_x.shape[0]):
        freq_order_x = freq_x/rps_x[frame]
        freq_order_y = freq_y/rps_y[frame] if not (x.equals(y)) else freq_order_x

        if rps_x[frame] == rps_y[frame]:
            Pxy[frame,:,:] = np.conjugate(fft_x[frame,:,:cols]) * fft_y[frame,:,:cols] * scale
            if not_divided_by_2:
                Pxy[frame,1:-1,:] *= 2
            else:
                Pxy[frame,1:,:] *= 2
            
            # updating dictionary
            for order in range(len(freq_order_x)):
                key = freq_order_x[order]
                if key in Pxy_dict:
                    Pxy_dict[key] = np.vstack([Pxy_dict[key], Pxy[frame, order, :]])
                else:
                    Pxy_dict.update({key: Pxy[frame, order, :]})
        else:
            for order in range(len(freq_order_x)):
                idx = binary_search(arr=freq_order_y, low=0, high=len(freq_order_y) - 1, x=freq_order_x[order])
                Pxy[frame, order, :] = np.conjugate(fft_x[frame, order, :cols]) * fft_y[frame, idx, :cols] * scale
                if not_divided_by_2 and order == len(freq_order_x) - 1:
                    print('nperseq not divided by 2')
                else:
                    Pxy[frame, order, :] *= 2
                # updating dictionary
                key = freq_order_x[order]
                if key in Pxy_dict:
                    Pxy_dict[key] = np.vstack([Pxy_dict[key], Pxy[frame, order, :]])
                else:
                    Pxy_dict.update({key: Pxy[frame, order, :]})
    # outputting for debug purpose
    #fft_frame_to_excel(Pxy, sheet_names=['x vs y'], fft_filename='Pxy.xlsx', index=freq_x)
        
    # averaging
    Pxy_averaged = pd.DataFrame(columns=x.columns[:cols])
    same_rps_fix_len_frame_bias = signal._spectral_py._median_bias(fft_x.shape[0])
    
    for freq_order in Pxy_dict.keys():
        if average == 'median':
            bias = signal._spectral_py._median_bias(Pxy_dict[freq_order].shape[0])
            if bias != same_rps_fix_len_frame_bias:
                print('bias %f != fbias %f'%(bias, same_rps_fix_len_frame_bias))
            if np.iscomplexobj(Pxy_dict[freq_order]):
                Pxy_averaged.loc[freq_order] = (np.median(np.real(Pxy_dict[freq_order]), axis=0) + 1j * np.median(np.imag(Pxy_dict[freq_order]), axis=0))
            else:
                Pxy_averaged.loc[freq_order] = np.median(Pxy_dict[freq_order], axis = 0)
            Pxy_averaged.loc[freq_order] /= bias
        elif average == 'mean':
            Pxy_averaged.loc[freq_order] = np.mean(Pxy_dict[freq_order], axis=0)
        else:
            raise ValueError('choose from specified methods')
        
    return Pxy_averaged.index, Pxy_averaged.to_numpy()

def coherence(x:pd.DataFrame, y:pd.DataFrame, fs:int, nperseg:int, noverlap:int, cols:int, domain:Literal['frequency', 'order'] = 'order', fg_column = 3,
              rps_x = None, rps_y = None, average:Literal['mean', 'median'] = 'mean'):
    '''
    rewrite signal.coherence adding domain parameter to get frequency as order of rotating frequency
    adding the average parameter for csd calculation 

    :param x: array_like. Time series of measurement values
    :param y: array_like. Time series of measurement values
    :param fs: float, optional. Sampling frequency of the x and y time series.
    :param nperseg: int. Length of each segment. length of the window.
    :param noverlap: int. Number of points to overlap between segments.
    :param cols: use first int(cols) for fft analysis, this values usually equal to the number of accelerometers.
    :param domain: units of the spectrum x labels
            'frequency': hz
            'order': relative to the inner race rotation, fr.
    :param fg_column: fg signal as square wave, usually the last column number of input data frame, unused if rps is given.
    :param rps_x: optional, round per second of each frame as a list with same length of frames of signal x.
    :param rps_y: optional, round per second of each frame as a list with same length of frames of signal y.
    :param average: { ‘mean’, ‘median’ },
                    Method to use when averaging periodograms. If the spectrum is complex, the average is computed separately for the real and imaginary parts. Defaults to ‘mean’.
    '''
    
    if domain == 'frequency':
        # same as signal.coherence but add param to change averaging method
        freqs, Pxx = signal.welch(x.iloc[:,:cols], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, axis=0, average=average)
        _, Pyy = signal.welch(y.iloc[:,:cols], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, axis=0, average=average)
        _, Pxy = signal.csd(x.iloc[:,:cols], y.iloc[:,:cols], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, axis=0, average=average)
        # organize into dataframe
        Cxy = pd.DataFrame(data=np.abs(Pxy)**2 / Pxx / Pyy, index=freqs)
        Cxy.index.rename('frequency [Hz]', inplace=True)
        x_limit = 5000
    else:
        # order of rotating frequency
        orders_x, Pxx = csd_order(x=x, y=x, fs=fs, nperseg=nperseg, noverlap=noverlap, cols=cols, fg_column=fg_column, rps_x=rps_x, rps_y=rps_x, average=average)
        orders_y, Pyy = csd_order(x=y, y=y, fs=fs, nperseg=nperseg, noverlap=noverlap, cols=cols, fg_column=fg_column, rps_x=rps_y, rps_y=rps_y, average=average)
        _, Pxy = csd_order(x=x, y=y, fs=fs, nperseg=nperseg, noverlap=noverlap, cols=cols, fg_column=fg_column, rps_x=rps_x, rps_y=rps_y, average=average)
        
        # the frequency order might not be the same, select the relative frequency order from original Pyy
        mapping_list = []
        for order in orders_x:
            idx = binary_search(orders_y, 0, len(orders_y) - 1, order)
            mapping_list.append(idx)
        
        Cxy = pd.DataFrame(np.abs(Pxy)**2 / np.real(Pxx) / np.real(Pyy)[mapping_list, :], index=orders_x)
        Cxy.index.rename('order of rotating frequency', inplace=True)
        x_limit = 25
    
    # visualize    
    fig, axs = plt.subplots(cols, 1, layout='constrained', sharex=True)
    
    column_name = ['%s vs %s'%(x.columns[i], y.columns[i]) for i in range(cols)]
    Cxy.columns = column_name
    for i in range(cols):
        ax = axs if cols == 1 else axs[i]
        Cxy.iloc[:,i].plot(ax=ax, legend=True, xlabel=Cxy.index.name, ylabel='Coherence', logx=False, logy=False, xlim=(0,x_limit))
        ax.grid(visible=True, which='both', axis='both')
        if i == 0:
            ax.set_title('Coherence (average type: %s)'%average)
    plt.show()
    
    return Cxy

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
    for i in range(len(good_sample_fft[benchmarks_sheet].index) - 1):
        # find the max of each type
        maximum = [0 for _ in range(len(types))]
        for sheet in good_sample_fft:
            idx_lwr = binary_search(good_sample_fft[sheet].index, 0, len(good_sample_fft[sheet].index) - 1, 
                                    good_sample_fft[benchmarks_sheet].index[i])
            idx_upr = binary_search(good_sample_fft[sheet].index, 0, len(good_sample_fft[sheet].index) - 1,
                                    good_sample_fft[benchmarks_sheet].index[i + 1])
            if i > 20 and i < 30: 
                print(idx_lwr, idx_upr)
            for col in range(len(good_sample_fft[sheet].columns)):
                for t in range(len(types)):
                    if good_sample_fft[sheet].columns[col].endswith(types[t]):
                        maximum[t] = max(np.max(good_sample_fft[sheet].iloc[idx_lwr:idx_upr, col]), maximum[t])
                        break
        # check each type is larger than the good samples or not
    '''
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
    '''
    
def plot_df_each_col_a_fig(df_dict:{pd.DataFrame}, types: list, axs: np.ndarray, **arg):
    for sheet in df_dict:
        for col_name in df_dict[sheet].columns:
            for i in range(len(types)):
                if col_name.endswith(types[i]):
                    axs[i].plot(df_dict[sheet].index, df_dict[sheet][col_name], **arg)
                    break

def fft_frame_to_excel(fft_frame:np.ndarray, sheet_names:list, fft_filename:str, index:np.ndarray):
    '''
    output a not averaged fft frames into excel, the shape of input frames should be [windowed_frame, frequency, column]
    the column >= 1 represnets the input channels
    '''
    # if file not exist
    if not os.path.exists(fft_filename):
        wb = openpyxl.Workbook()
        wb.save(fft_filename)
        wb.close()
    
    for sheet in range(len(sheet_names)):
        df_fft = pd.DataFrame(fft_frame[:,:,sheet].transpose(), index=index, columns=range(fft_frame.shape[0]))
        with pd.ExcelWriter(fft_filename, mode="a", if_sheet_exists="new", engine="openpyxl") as writer:
            df_fft.to_excel(writer, sheet_name=sheet_names[sheet])
            if 'Sheet' in writer.book.sheetnames:
                writer.book.remove(writer.book['Sheet'])
            writer.book.save(fft_filename)
            writer.book.close()

def coherence_test(average:Literal['mean', 'median'] = 'mean', visualize_fig:int = 1):
    np.random.seed(0)
    fs = 51200
    frame_len = 8192
    n = 8192*5
    t = np.arange(n)/fs
    f1 = 40.75
    f2 = 41.75
    
    x = 10 * np.sin(2 * np.pi * f1 * t) + 10 * np.sin(2 * np.pi * 2 * f1 * t) + 10 * np.sin(2 * np.pi * 3 * f1 * t) + 0.5 * np.random.randn(n)
    y = 15 * np.sin(2 * np.pi * f2 * t + np.pi / 4) + 10 * np.sin(2 * np.pi * 2 * f2 * t) + 10 * np.sin(2 * np.pi * 3 * f2 * t) + 0.5 * np.random.randn(n)
    nframe = int(n / (frame_len/4) - (4 - 1))
    freq_csd, Pxy = csd_order(x=pd.DataFrame(x, columns=['x']), y=pd.DataFrame(y, columns=['y']), fs=fs, nperseg=frame_len, noverlap=frame_len*0.75, cols=1,
                              rps_x=[43.75 for i in range(nframe)], rps_y=[43.75 for i in range(nframe)], average=average)
    if visualize_fig==1:
        freq_csd2, Pxy2 = signal.csd(x, y, fs, nperseg=frame_len, noverlap=frame_len*0.75, average=average)
        # debug
        freq_x, fft_x, rps_x = fft(df=pd.DataFrame(x, columns=['x']), fs=fs, nperseq=frame_len, window=np.hanning(frame_len), noverlap=0.75*frame_len, rps=[43.75 for i in range(nframe)])
        freq_y, fft_y, rps_y = fft(df=pd.DataFrame(y, columns=['y']), fs=fs, nperseq=frame_len, window=np.hanning(frame_len), noverlap=0.75*frame_len, rps=[43.75 for i in range(nframe)])

        fig, axs = plt.subplots(4, 1, layout='constrained')
        
        axs[0].plot(freq_x, np.abs(fft_x[0, :, :]))
        axs[0].set_xlim(0, 500)
        axs[0].set_yscale('log')
        axs[0].set_title('input signal x fft spectrum at first frame, base freq = %.2f'%f1)
        axs[0].set_xlabel('frequency (Hz)')
        axs[1].plot(freq_y, np.abs(fft_y[0, :, :]))
        axs[1].set_xlim(0, 500)
        axs[1].set_yscale('log')
        axs[1].set_xlabel('frequency (Hz)')
        axs[1].set_title('input signal y fft spectrum at first frame, base freq = %.2f'%f2)
        axs[2].plot(freq_csd2, np.abs(Pxy2))
        axs[2].set_xlim(0, 500)
        axs[2].set_yscale('log')
        axs[2].set_xlabel('frequency (Hz)')
        axs[2].set_title('original csd')
        axs[3].plot(freq_csd, np.abs(Pxy))
        axs[3].set_xlim(0, 12)
        axs[3].set_yscale('log')
        axs[3].set_xlabel('order')
        axs[3].set_title('order csd')
        # annotate peak
        annotatePeaks(x=freq_x, y=np.abs(fft_x[0, :, 0]), ax= axs[0], prominence=10e2, rotation = 0,
                    annotateY=True, dot = 'x', xytext=(0, 0), arrowprops=None)
        annotatePeaks(x=freq_y, y=np.abs(fft_y[0, :, 0]), ax= axs[1], prominence=10e2, rotation = 0,
                    annotateY=True, dot = 'x', xytext=(0, 0), arrowprops=None)
        annotatePeaks(x=freq_csd2, y=np.abs(Pxy2), ax= axs[2], prominence=10e-3, rotation = 0,
                    annotateY=True, dot = 'x', xytext=(0, 0), arrowprops=None)
        annotatePeaks(x=freq_csd, y=np.abs(Pxy[:,0]), ax= axs[3], prominence=10e-3, rotation = 0,
                    annotateY=True, dot = 'x', xytext=(0, 0), arrowprops=None)
        
        plt.show()

    if visualize_fig==2:
        #freq_cohere, Cxy = signal.coherence(x, y, fs, nperseg=frame_len, noverlap=frame_len*0.75)
        Cxy = coherence(x=pd.DataFrame(x, columns=['x']), y=pd.DataFrame(y, columns=['y']), 
                        fs=fs, nperseg=frame_len, noverlap=frame_len*0.75, cols=1, domain = 'order', 
                        rps_x=[43.75 for i in range(nframe)], rps_y=[43.75 for i in range(nframe)], average=average)
        freq_cohere = Cxy.index
        fig, axs = plt.subplots(3, 1, layout='constrained')
        # input
        axs[0].plot(t, x, label='%.1f Hz base freq'%f1)
        axs[0].plot(t, y, label='%.1f Hz base freq'%f2)
        axs[0].set_xlabel('time(s)')
        axs[0].set_ylabel('input signal')
        axs[0].set_xlim(0, 0.25)
        axs[0].legend()
        # csd
        axs[1].semilogy(freq_csd, np.abs(Pxy))
        axs[1].set_xlabel('Frequency [Hz]')
        axs[1].set_ylabel('CSD [V**2/Hz]')
        axs[1].set_xlim(0, 25)
        # coherence
        axs[2].semilogy(freq_cohere, Cxy) 
        axs[2].set_xlabel('Frequency [Hz]')
        axs[2].set_ylabel('Coherence')
        axs[2].set_xlim(0, 25)
        plt.show()

def time_dependent_coherence(x: pd.Series, y:pd.Series):
    s1 = TimeSeries(x, sample_rate=51200)
    s2 = TimeSeries(y, sample_rate=51200)
    coh = s1.coherence_spectrogram(s2, stride=0.5, fftlength=0.5, overlap=0.25)
    df = pd.DataFrame(data=coh.zip())
    df.to_excel('spectrogram.xlsx')
    #coh = s1.coherence(s2, fftlength=8192/51200, overlap=8192*0.75/51200, window='flattop')
    plot = coh.plot(xlabel='Frequency [Hz]', #xscale='log',
                    ylabel='Coherence', #yscale='linear',
                    title='Coherence between acc data of %s and %s'%(x.name, y.name))
    ax = plot.gca()
    ax.grid(True, 'both', 'both')
    ax.colorbar(label='Coherence', clim=[0,2], cmap='plasma')
    plot.show()

if __name__ == '__main__':
    coherence_test(average='median')
    #normal_df = pd.read_excel('../../test_data//Defective_products_on_line_20%//acc_data//000045_lr.xlsx', header=0, usecols="A:D")
    #abnormal_df = pd.read_excel('../../test_data//20240911_good_samples//acc_data//000039_lr.xlsx', header=0, usecols="A:D")
    #coherence(x=normal_df, y=abnormal_df, fs=51200, nperseg=8192, noverlap=8192*0.75, cols=3, domain='order', average='mean')
    #normal_fft_df = fft_processing('../../test_data//20240911_good_samples//fft.xlsx', usecols=[0,1,2,3], combine=True)
    #abnormal_fft_df = fft_processing('../../test_data//Defective_products_on_line_20%//fft_abnormal.xlsx', usecols=[0,1,2,3], combine=True)
    #df_fft = pd.concat([normal_fft_df, abnormal_fft_df], axis=1)
    #distvec = scipy.spatial.distance.pdist(df_fft.iloc[268:377,:].transpose(), metric='cosine') # most sensitive range 2000-5000hz
    #m = scipy.spatial.distance.squareform(distvec)
    #matrix = pd.DataFrame(1-m, index=df_fft.columns, columns=df_fft.columns)
    #matrix.to_excel('cosine_similarity.xlsx')