# Vibration analysis

## Data format

### accelerometer signal sequence

This is a time domain measurement row data in unit g(m/s^2), the data length is defined in `analog_input.py`.
Each measurement is store in a `.xlsx` file with single sheet to prevent large file size. The file is saved with only top row of header.

### FFT file

All measurement that after FFT will saved in different sheet of the same `.xlsx` file, with index in the first column, representing the frequency or the order of rotation frequency. The order of rotation frequency is used for compatibility of different rotation speed of sample.

## Machine learning
### Anomaly detection
#### Integrate MLflow into current training workflow 
Workflow Summary

* **Preparation (One-time)**
Ensure rclone is installed and configured with a remote (e.g., google_drive_mlflow) pointing to your Google Drive.

* **Mount**:

Crucial: Before running `mount_gdrive.sh`, ensure the local `mlruns` directory in your project is empty or doesn't exist. If it exists and contains data from a previous run, it will prevent correct mounting.

Execute:
```sh        
bash mount_gdrive.sh ./mlruns (or your chosen mount point). #This will mount your Google Drive's mlruns subdirectory to your local ./mlruns folder.
```
* **Run MLflow**:

In your Python script, set the MLflow tracking URI: 
```python
mlflow.set_tracking_uri("file://./mlruns"). # or chosen directory
```
Run your Python script to train models and log results. 
> If the mount directory does not exist because the **Mount** step does not execute successful, a new directory will be created, this directory can not set as mount point because it will conflict with`rclone`.

* **(Optional)** In a separate terminal, start the MLflow UI: 
```sh
mlflow ui
``` 
(this will read from the mounted `mlruns` directory).

* **Unmount**:
Once your Python script has finished logging, execute: 
```sh
bash unmount_gdrive.sh ./mlruns.
```
This will unmount the Google Drive and then delete the local `./mlruns` directory (the mount point itself), ensuring it's clean for the next mount operation.

## Reference

### Machine Learning

#### Auto-sklearn
https://andy6804tw.github.io/crazyai-ml/20.Auto-Sklearn/

#### Support vector machine

* [Faulty bearing detection, classification and location in a three-phase induction motor based on Stockwell transform and support vector machine](https://www.sciencedirect.com/science/article/abs/pii/S0263224118308418)
* [Classifying data using the SVM algorithm using Python](https://developer.ibm.com/tutorials/awb-classifying-data-svm-algorithm-python/)

#### Similarity
* https://ithelp.ithome.com.tw/articles/10268777
* https://tomhazledine.com/cosine-similarity-alternatives/

#### fisher score
Install [scikit-feature](https://github.com/jundongl/scikit-feature)
```sh
pip install git+https://github.com/jundongl/scikit-feature.git
```
[Examples](https://blog.csdn.net/qq_39923466/article/details/118809782)


#### Shapley Value
Install [shap](https://pypi.org/project/shap/)
https://www.analyticsvidhya.com/blog/2019/11/shapley-value-machine-learning-interpretability-game-theory/


### Data processing

#### Correlation

* [Correlation Coefficient](https://mathworld.wolfram.com/CorrelationCoefficient.html)
* [Correlation-based Feature Selection in Python from Scratch](https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/)
* [pair variable plot to visualize correlation](https://stackoverflow.com/questions/55113349/how-to-calculate-the-correlation-of-all-features-with-the-target-variable-binar)

#### Frequency transform

* [PSD, RMS and other data analysis index](https://info.endaq.com/hubfs/Plots/enDAQ-Vibration-Monitoring-Metrics-Case-Western-Bearing-Data_2.html#Appendix)
* [Top 12 Vibration Metrics to Monitor & How to Calculate Them](https://blog.endaq.com/top-vibration-metrics-to-monitor-how-to-calculate-them)
* [PSDs are preferred to FFTs for vibration analysis](https://blog.endaq.com/why-the-power-spectral-density-psd-is-the-gold-standard-of-vibration-analysis)
* [Head-acoustics FFT analysis](https://cdn.head-acoustics.com/fileadmin/data/global/Application-Notes/SVP/FFT-Octave-Analysis-Wavelet-02.2018.pdf)

#### Weighting

* [ABC_weighting.py](https://github.com/endolith/waveform-analysis/blob/master/waveform_analysis/weighting_filters/ABC_weighting.py)
* [Using A-Weighting on time signal](https://stackoverflow.com/questions/65842795/using-a-weighting-on-time-signal)

#### Empirical Mode Decomposition

* [Tutorial and code example](https://www.youtube.com/watch?v=eiqsAGlAPYY&list=PLkoI-nNk12tsAiwQ1vdkHUVzsf0KvzOf_&index=5)
* MEMD library for python: [MEMD-Python-](https://github.com/mariogrune/MEMD-Python-/tree/master)
* [python library: emd](https://emd.readthedocs.io/en/stable/index.html)
* [python library: PyEMD](https://pyemd.readthedocs.io/en/latest/search.html?q=multivariate&check_keywords=yes&area=default#)
    >[tutorial](https://zhuanlan.zhihu.com/p/581693595)

#### Wavelet Decomposition

* [Motor Fault Diagnosis Algorithm Based on Wavelet and Attention Mechanism](https://onlinelibrary.wiley.com/doi/10.1155/2021/3782446#abstract)
* [Motor Fault Detection Using Wavelet Transform and Improved PSO-BP Neural Network](https://www.mdpi.com/2227-9717/8/10/1322)
* Jiménez, G., Muñoz, A. & Duarte-Mermoud, M. Fault detection in induction motors using Hilbert and Wavelet transforms. Electr Eng 89, 205–220 (2007). https://doi.org/10.1007/s00202-005-0339-6
* [Induction motor fault detection based onmulti-sensory control and wavelet analysis](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/iet-epa.2020.0030)
* [Diagnostics Of Faults In Induction Motor Via Wavelet Packet 
Transform](https://www.iosrjournals.org/iosr-jvlsi/papers/Conf-ICETETR-2016/Volume%201/1.%2001-06.pdf)
* [Bearing fault detection in a 3 phase induction motor using stator current frequency spectral subtraction with various wavelet decomposition techniques](https://www.sciencedirect.com/science/article/pii/S2090447917300771)
* [FPGA based embedded system for induction motor failure monitoring at the start-up transient vibrations with wavelets](https://ieeexplore.ieee.org/document/4577701)

#### Variational Mode Decomposition

* [researches with this keyword](https://www.sciencedirect.com/topics/engineering/variational-mode-decomposition#:~:text=Variational%20mode%20decomposition%20%28VMD%29%20is%20the%20latest%20signal,detection%20method%20is%20reported%20in%20Ref.%20%5B61%20%5D.)

### Data Acquisition

#### NI C library

* [NIDAQ c library](https://www.ni.com/en/support/documentation/supplemental/21/using-ni-daqmx-in-text-based-programming-environments.html)
* [NI-DAQ™mx C Reference](https://www.ni.com/docs/zh-TW/bundle/ni-daqmx-c-api-ref/page/cdaqmx/related_documentation.html)
* [vibration analysis instructions](https://hecoinc.com/the-importance-of-route-based-data-acquisition-series/)
* [vibration DAQ System](https://dataloggerinc.com/data-acquisition-systems/vibration-daq-systems/)
* vscode settings and reference material for NI Linex Real_Time OS: [link](https://github.com/edavis0/nidaqmx-c-examples?tab=readme-ov-file)

#### NI Python Library

* [NIDAQ Python github](https://github.com/ni/nidaqmx-python/blob/master/examples/counter_in/read_freq.py)
* [nidaqmx documentation](https://nidaqmx-python.readthedocs.io/en/latest/task_channels.html#nidaqmx.task.channels.CIChannel.ci_freq_term_cfg)

### ball bearing vibration analysis

1. C. Rodriguez-Donate, R. J. Romero-Troncoso, A. Garcia-Perez and D. A. Razo-Montes, "FPGA based embedded system for induction motor failure monitoring at the start-up transient vibrations with wavelets," 2008 International Symposium on Industrial Embedded Systems, Le Grande Motte, France, 2008, pp. 208-214, doi: 10.1109/SIES.2008.4577701. keywords: {Field programmable gate arrays;Embedded system;Induction motors;Condition monitoring;Vibration measurement;Discrete wavelet transforms;Electrical equipment industry;Electrical products industry;Machinery production industries;Performance evaluation;Vibration analysis;induction motor monitoring;embedded system;FPGA;wavelets}, [link](https://ieeexplore.ieee.org/document/4577701) 
2. I. A. Jamil, M. I. Abedin, D. K. Sarker and J. Islam, "Vibration data acquisition and visualization system using MEMS accelerometer," 2014 International Conference on Electrical Engineering and Information & Communication Technology, Dhaka, Bangladesh, 2014, pp. 1-6, doi: 10.1109/ICEEICT.2014.6919090. keywords: {Vibrations;Accelerometers;Vibration measurement;Acceleration;Software;Micromechanical devices;Microcontrollers;C# .NET;Data Acquisition;Serial Communication;Real-time Visualization;Mechanical Vibration Measurement}, [link](https://ieeexplore.ieee.org/document/6919090)
3. Shrivastava, Amit & Wadhwani, Sulochana. (2012). Vibration signature analysis for Ball Bearing of Three Phase Induction Motor. IOSR Journal of Electrical and Electronics Engineering. 1. 10.9790/1676-0134650. [PDF](https://www.iosrjournals.org/iosr-jeee/Papers/vol1-issue3/G0134650.pdf)
4. M. S. Moiz et al., "Health Monitoring of Three-Phase Induction Motor Using Current and Vibration Signature Analysis," 2019 International Conference on Robotics and Automation in Industry (ICRAI), Rawalpindi, Pakistan, 2019, pp. 1-4, doi: 10.1109/ICRAI47710.2019.8967356. keywords: {Current signature analysis;Vibration signature analysis;spectral analysis;stator current;preventive maintenance}, [link](https://ieeexplore.ieee.org/document/8967356)
