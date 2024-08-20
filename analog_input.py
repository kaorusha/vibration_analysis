import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE, EncoderType
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai0", "acc0", sensitivity=10.22)
    task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai1", "acc1", sensitivity=10.11)
    task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai2", "acc2", sensitivity=10.28)
    #task.ci_channels.add_ci_freq_chan use this https://github.com/ni/nidaqmx-python/blob/master/examples/counter_in/read_freq.py
    task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai3", "fg", sensitivity=600)
    fs = 51200
    acq_time = 10
    task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=fs*acq_time)
    t1 = time.perf_counter()
    data = task.read(READ_ALL_AVAILABLE)
    t2 = time.perf_counter()
    print('acquire time: {}'.format(t2 - t1))
    file_name = '001833_ud'
    df = pd.DataFrame({file_name+'_up':data[0], file_name+'_down':data[1], file_name+'_axial':data[2], file_name+'fg':data[3]})
    df.plot(title='acc data', xlabel='time(sec)', ylabel='g')
    df.to_excel('../%s.xlsx'%file_name, sheet_name='acc_data', index=False)
    plt.show()