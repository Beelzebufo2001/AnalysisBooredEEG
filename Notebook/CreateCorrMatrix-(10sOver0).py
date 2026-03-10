import os
import mne
import numpy as np
import warnings as w
import matplotlib.pyplot as plt
from scipy.signal import decimate
from elephant.signal_processing import butter

#sample_name = "C0"
#file_name = "C01_2024-11-16_10-44-51_RS_before_ica.fif"

data_info = input("Enter: sample_name file_name\n").split()

if len(data_info) < 2:
    print("Need sample_name and filename")
    exit()

sample_name = data_info[0]
file_name = data_info[1]

path = "./../data/"+ sample_name + "/" + file_name

out_dir = "Corr_matrices/" + sample_name
os.makedirs(out_dir, exist_ok=True)

with w.catch_warnings():
    w.simplefilter("ignore", RuntimeWarning)
    ica = mne.io.read_raw_fif(path, preload=False)
    
#Setting montage
ica.set_montage(mne.channels.make_standard_montage("standard_1005"))

if input("are you sure? (y/n)") != "y":
    exit()

#--------Frequency--------
old_freq = int(ica.info['sfreq'])
new_freq = 250

#--------Samples size--------
len_snipp = 10 #len of snipplet
n_samples = ica.n_times
window_samples = int(len_snipp*old_freq)

#--------While cycle--------
t_zero = 0 #start
maximal = 0 #Limit size

existing = len(os.listdir(out_dir))
t_zero = existing * old_freq

while(t_zero+window_samples <= n_samples): #Make cycle over 10s in 1 sec jump
    snipplet = ica.get_data(#create snipplets
        start=t_zero, 
        stop=(t_zero+window_samples)
    )
    
    """
    Butter your way to 1–45 Hz 
    """
    quatro_s = butter( 
        snipplet,
        highpass_frequency=1,
        lowpass_frequency=45,
        filter_function='sosfiltfilt',
        sampling_frequency=old_freq, 
        axis=1, order=6
    )
   
    """
    Downsample (8000 -> 250 Hz)
    """
    down_quatro_s = decimate(
        quatro_s,
        q = int(old_freq/new_freq),
        ftype='fir',
        axis=1,
        zero_phase=True #to prevent phase shift:: 
    )
    
    """
    Correlation Matrix
    """
    matrix = np.corrcoef(down_quatro_s)
    
    """
    Save Matrix
    """
    np.save(
        os.path.join(out_dir, f"corr_{int(t_zero // old_freq):04d}.npy"),
        matrix
    )

    #--------Change variable--------
    t_zero += old_freq

