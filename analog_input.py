import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE, EncoderType
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time

def accelerometer_input():
    '''Acquire accelerometer data from NI cDAQ-9174 with NI 9234 module.'''
    if len(sys.argv) != 2 or sys.argv[1] not in ['1', '2']:
        print('Usage: python analog_input.py <direction>\n<direction> 1: left-right, 2: up-down')
        sys.exit(1)
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai0", "acc0", sensitivity=10.22) # serial no: 506751
        task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai1", "acc1", sensitivity=10.11) # serial no: 506750
        task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai2", "acc2", sensitivity=10.28) # serial no: 506773
        #task.ci_channels.add_ci_freq_chan use this https://github.com/ni/nidaqmx-python/blob/master/examples/counter_in/read_freq.py
        task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai3", "fg", sensitivity=600)
        fs = 51200
        acq_time = 10
        task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=fs*acq_time)
        t1 = time.perf_counter()
        data = task.read(READ_ALL_AVAILABLE)
        t2 = time.perf_counter()
        print('acquire time: {}'.format(t2 - t1))
        t3 = time.perf_counter()
        dir = '../../test_data//20250613_test_samples//acc_data_100%//'
        file_name = 'b04802'
        file_name += '_lr' if sys.argv[1] == '1' else '_ud'
        if file_name.endswith('lr'):
            df = pd.DataFrame({file_name+'_left':data[0], file_name+'_right':data[1], file_name+'_axial':data[2], file_name+'_fg':data[3]})
        if file_name.endswith('ud'):
            df = pd.DataFrame({file_name+'_up':data[0], file_name+'_down':data[1], file_name+'_axial':data[2], file_name+'_fg':data[3]})
        df.to_parquet(dir + '%s.parquet.gzip'%file_name, compression='gzip', index=False)
        t4 = time.perf_counter()
        print('write time: {}'.format(t4 - t3))
        df.iloc[:,0:3].plot(title='acc data', xlabel='time(sec)', ylabel='g')
        plt.show()

def microphone_input():
    '''Acquire microphone data from NI cDAQ-9174 with NI 9234 module.'''
    with nidaqmx.Task() as task:
        task.ai_channels.add_teds_ai_microphone_chan("cDAQ1Mod1/ai0", "mic0", max_snd_press_level=146, current_excit_val=0.2)
        task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai1", "fg")
        fs = 51200
        acq_time = 1
        task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=fs*acq_time)
        t1 = time.perf_counter()
        data = task.read(READ_ALL_AVAILABLE)
        t2 = time.perf_counter()
        print('acquire time: {}'.format(t2 - t1))
        t3 = time.perf_counter()
        dir = '../../test_data//20250620_test_samples//mic_data_test//'
        file_name = 'b04802_mic'
        df = pd.DataFrame({file_name+'_mic': data[0], file_name+'_fg': data[1]})
        df.to_parquet(dir + '%s.parquet.gzip'%file_name, compression='gzip', index=False)
        t4 = time.perf_counter()
        print('write time: {}'.format(t4 - t3))
        df.plot(title='mic data', xlabel='time(sec)', ylabel='Pa')
        plt.show()

if __name__ == '__main__':
    microphone_input()