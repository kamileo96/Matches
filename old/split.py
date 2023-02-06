import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = './myaudio/test2-45z.wav'
signal, sr = librosa.load(file, sr=2*22050)
signal = np.array(signal)
#librosa.display.waveshow(signal, sr=sr)
#plt.show()
plt.plot(np.arange(len(signal))/sr,signal)
plt.show()
exit()
asignal = np.abs(signal)
avsize = 100
def signal_average(asignal, avsize, hop):
    return np.sum(np.array([np.roll(asignal, i*hop) for i in range(-avsize,avsize+1)]), axis=0)/(2*avsize + 1)
signal_avg = signal_average(asignal, avsize, 1)


detect_min = 0.01
impuls_max_time = 0.3 #s
#minimal interval for the program to work

det_peaks_idx = np.empty(0, dtype=int)
search = True

for i, s in enumerate(signal_avg):
    if search and s > detect_min:
        #print(f'found i {i} s {s}')
        search = False
        ifin = i + impuls_max_time*sr
        maxs = s
        maxsidx = i
    if not search:
        if s > maxs: 
            maxs = s
            maxsidx = i
        if i > ifin:
            det_peaks_idx = np.append(det_peaks_idx, maxsidx)
            search = True
if len(det_peaks_idx) == 0: 
    print('no peaks found. try changing detect_min')


#this is for manualy detecting left and right boundaries of signal
#i have instead decided to set a constant signal length,
#under the assumntion the signals are separated enough.
"""
signal_avg2 = signal_average(signal_avg, avsize, 2)
signal_avg3 = signal_average(signal_avg2, avsize, 1)
noise_level = 0.0017
potential_boundaries = np.nonzero((np.roll(signal_avg3, -1) > signal_avg3)*(signal_avg3 < np.roll(signal_avg3, 1))*(signal_avg3<noise_level))[0]
segment_data = np.empty((len(det_peaks_idx),4),dtype=int)
for i,peak_idx in enumerate(det_peaks_idx):
    idxsorted = np.searchsorted(potential_boundaries, peak_idx)
    left = potential_boundaries[idxsorted - 1]
    right = potential_boundaries[idxsorted]
    if (right - left)/sr > impuls_max_time*3: print(f'error: too long segment {i} : {(right - left)/sr}')
    segment_data[i] = np.array([left, right, 0, peak_idx]) # 0 is placeholder for left peak

plt.plot(det_peaks_idx, signal_avg3[det_peaks_idx], 'o')
plt.plot(segment_data.T[0], signal_avg3[segment_data.T[0]], 'o')
plt.plot(segment_data.T[1], signal_avg3[segment_data.T[1]], 'o')

print(segment_data)
plt.plot(signal_avg3)
plt.show()
"""

#impuls: <---delta---impuls_time---delta--->
delta = 0.2 #s
impuls_time = 0.11
dleft = (impuls_time + delta)*sr
dright = delta*sr
segment_data = np.array([np.array([p - dleft, p + dright, 0, p]) for p in det_peaks_idx]) # 0 is placeholder for left peak

zer=np.zeros(len(det_peaks_idx))
plt.plot(det_peaks_idx/sr, zer, 'o')
plt.plot(segment_data.T[0]/sr, zer, 'o')
plt.plot(segment_data.T[1]/sr, zer, 'o')
plt.plot(det_peaks_idx/sr - impuls_time, zer, 'o')
plt.show()

plt.plot(signal_avg)
plt.plot(det_peaks_idx, signal_avg[det_peaks_idx], 'o')
plt.show()