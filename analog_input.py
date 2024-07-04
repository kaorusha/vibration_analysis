import nidaqmx
from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
import matplotlib.pyplot as plt
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai0", "acc0")
    task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai1", "acc1")
    task.ai_channels.add_ai_accel_chan("cDAQ1Mod1/ai2", "acc2")
    task.timing.cfg_samp_clk_timing(48000, sample_mode=AcquisitionType.FINITE, samps_per_chan=10)
    data = task.read(READ_ALL_AVAILABLE)
    plt.plot(data[0])
    plt.plot(data[1])
    plt.plot(data[2])
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()