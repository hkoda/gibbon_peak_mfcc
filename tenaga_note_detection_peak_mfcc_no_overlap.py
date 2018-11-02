# coding:utf-8

'''
Main code for for the mfcc_s at the local peak of song frame data.
For voice activity detection (VAD), a GitHub repo code for VAD (available in GitHub repo in https://github.com/marsbroshok/VAD-python) was used.
For mfcc values, python_speech_features (available in GitHub repo, https://github.com/jameslyons/python_speech_features, and installed by pip install python_speech_features) were used.
'''

import glob
import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
import copy
from vad import VoiceActivityDetector
from scipy import signal
from fractions import Fraction
import pandas as pd
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

def data2hilbert(data,fs):
    # make list of cut-off frequencies for bandpass filter.
    cut_off_freq = [80,260,600,1240,2420,4650,7999] # applied from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2268248/.
    # upper bound of the wave data sampled at 16000 Hz is 8000 Hz; therefore upper limit is set as 8000 - 1 Hz, here.
    data_passed_hilbert_sum = np.array([])
    for i in range(len(cut_off_freq)-1):
        data_passed = band_pass_filter(data,fs,cut_off_freq[i],cut_off_freq[i+1])
        data_passed_hilbert = abs(signal.hilbert(data_passed))
        data_passed_hilbert_sum = data_passed_hilbert
        if not i:
            data_passed_hilbert_sum += data_passed_hilbert
    return data_passed_hilbert_sum

def split_frame_hilbert_peak(dw,st,data_hilbert):
    # First, computing the simple moving average by n samples window length. n = 320 samples: 320 / 16000 = 0.02 sec
    data_hilbert_sma = np.convolve(data_hilbert,np.ones(1600)/1600,"same")
    # make np array stacking the time data of the note.
    note_time = np.array([[0],[0],[0]],dtype = float)
    for i in range(len(st)):
        if st[i]["speech_begin_sample"] != 0: # pass if na included in dictionary list.
            start_t = st[i]["speech_begin_sample"]
            end_t = st[i]["speech_end_sample"]
            start_pos = np.where(dw == int(start_t))[0] / 100.0 * 16000
            end_pos = np.where(dw == int(end_t))[0] / 100.0 * 16000
            data_hilbert_phrase = data_hilbert_sma[int(start_t):int(end_t) + 800]
            pk_s = find_peaks(data_hilbert_phrase,1,1600*3,160,2400,1600)
            pk_s_label = np.array([1]* len(pk_s[0]))
            pk_s_phrasw_id = np.array([i+1]*len(pk_s[0]))
            pk_s_3d = np.vstack(((pk_s[0]+ int(start_t))/16000.0,pk_s_label,pk_s_phrasw_id))
            note_time = np.hstack((note_time,pk_s_3d))
    note_time = np.delete(note_time,np.where(np.isnan(note_time[0])),axis =1)
    note_time = np.hstack((note_time,np.array([[len(data_hilbert) / 16000.0],[0],[0]]))) # add last time
    return note_time

def band_pass_filter(data,fs,freq_lower,freq_upper):
    # initialize parameter setting
    nyq = fs / 2.0 # nyquist frequency
    # make bandbasss filter
    fe1 = freq_lower / nyq
    fe2 = freq_upper / nyq
    numtaps = 225 # filter coefficients, needs odds
    if fe1 > 0:
        b = signal.firwin(numtaps,[fe1,fe2],pass_zero=False)
    else:
        b = signal.firwin(numtaps,[fe2]) # simply lowpass filter including 0 Hz
    # apply FIR filter to signal
    data_passed = signal.lfilter(b,1,data)

    return data_passed

def find_peaks(a,local_height_thre,local_width,step,min_peak_distance,search_min_distance):
    # a: sequence data for finding local peaks
    # local_height_thre: local height threshold.
    # local width: window width to find local peaks. Local peak shoud be located as inverse-U shape portion of this length window. 
    # step: search steps.
    # min_peak_distance: at least, local peaks should be spaced over this distance. 
    # search_min_distance: local minimum should be searched from local peak location - this search_min_distance to local peak location.
    last_peak_index = 0
    result_pk_idxs = []
    result_vally_idxs = []
    nframes = (len(a) - local_width)/step
    for i in range(nframes):
        start_index = i * step
        end_index = start_index + local_width
        local_peak = np.max(a[start_index:end_index])
        local_peak_index = np.where(a == local_peak)[0]
        if (local_peak_index > start_index and local_peak_index < end_index) and (local_peak_index - last_peak_index > min_peak_distance):
            local_min_left = np.min(a[start_index:local_peak_index[0]])
            local_height_left = local_peak - local_min_left
            local_min_right = np.min(a[local_peak_index[0]:end_index])
            local_height_right = local_peak - local_min_right
            if (local_height_left > local_height_thre) and (local_height_right > local_height_thre):
                result_pk_idxs.append(local_peak_index[0])
                local_valley = np.min(a[local_peak_index[0]-search_min_distance:local_peak_index[0]])
                local_min_left_index = np.where(a == local_valley)[0]
                result_vally_idxs.append(local_min_left_index[0])
                last_peak_index = local_peak_index
            if (local_peak_index - last_peak_index < min_peak_distance) and (a[local_peak_index] > a[last_peak_index]):
                result_pk_idxs[-1] = local_peak_index
                local_valley = np.min(a[local_peak_index[0]-search_min_distance:local_peak_index[0]])
                local_min_left_index = np.where(a == local_valley)[0]
                result_vally_idxs[-1] = local_min_left_index[0]
                last_peak_index = local_peak_index
    result_pk_idxs = np.array(result_pk_idxs)
    if (result_vally_idxs and result_vally_idxs[0] < search_min_distance * 2):
        result_vally_idxs[0] = 0
    result_vally_idxs = np.array(result_vally_idxs)
    if result_pk_idxs.any():
        result_pk_value = a[result_pk_idxs]
        result_vally_value = a[result_vally_idxs]
    else:
        result_pk_idxs = np.array([0])
        result_pk_value = a[result_pk_idxs]
        result_vally_idxs = np.array([0])
        result_vally_value = a[result_vally_idxs]
    return (result_pk_idxs, result_pk_value,result_vally_idxs,result_vally_value)

def get_mfcc_s(a,fs,start_time,end_time,frames=0.025,overlap=0.01):
    start_pos = int(start_time * fs)
    end_pos = int(end_time * fs)
    mfcc_s = mfcc(a[start_pos:end_pos],samplerate = fs,winlen=frames,winstep=overlap,numcep=13)
    return mfcc_s

def mfcc2str(mfcc_s):
    mfcc_s_str = '\t'.join([','.join(map(str,mfcc_s[i].tolist())) for i in range(len(mfcc_s))])
    return mfcc_s_str

def main():
    fn_s = glob.glob('./wav/171219101741/*.wav') # set the wave data directory.
    now = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    save_path = os.path.join('./note_annotation', now)
    os.makedirs(save_path)
    for fn_full in fn_s:
        fn,ext = os.path.splitext(fn_full)
        v = VoiceActivityDetector(fn+ext)
        dw, sve, pe = v.detect_speech() # dw: detected windows; sve: sum_voice_energy_s; pe: peak energy 
        st = v.convert_windows_to_readible_labels(dw) # return the list of speech times based on the detected window lists.
        data_passed_hilbert_sum = data2hilbert(v.data,v.rate) # hilbert transform
        note_time = split_frame_hilbert_peak(dw,st,data_passed_hilbert_sum)
        # here set frame length for putting in mfcc compute 0.1
        df_note_time = pd.DataFrame(
                                    np.vstack((note_time[0][:],note_time[0][:]+0.1,note_time[2][:])).T,
                                    columns = ["start_time","end_time","phrase_label"]
                                    )
        df_note_time_sel = df_note_time[df_note_time.phrase_label > 0]
        df_note_time_sel.reset_index()
        df_note_time_sel.to_csv(save_path+'/'+fn.split('/')[-1] +'_phrase_peak_time.csv')
        mfcc_s_str_100_wo = ''
        for i in range(len(df_note_time_sel.index)):
            mfcc_s_100 = get_mfcc_s(v.data,v.rate,df_note_time_sel['start_time'].iloc[i],df_note_time_sel['end_time'].iloc[i],frames=0.1,overlap=0.1)
            mfcc_s_str_100_wo += mfcc2str(mfcc_s_100)+'\n'
        with open(save_path+'/'+fn.split('/')[-1]+'_100msec_no_overlap_at_peak_loc.ghmm', 'w') as f_samples_100:
    		f_samples_100.write(mfcc_s_str_100_wo)
        print('done for %s' % fn)

if __name__ == '__main__':
    main()
