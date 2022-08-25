##############
# 输入：经过TQWT处理的SEED第一天的数据
# 输出：62个通道8个频段的DE特征,存储为 .mat
##############

import os
import sys
import math
import numpy as np
# import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from scipy.fftpack import dct
from scipy.fftpack import fft
import h5py
from scipy.stats import skew
from scipy.stats import kurtosis


def decompose(file, name):
    # trial*channel*sample
    data = h5py.File(file+'.mat')
    frequency = 200

    eeg_data = {}
    decomposed_de = np.empty([0, 63, 8])

    for trial in range(15):

        tmp_trial_signal = data['sub_tqwt'][name + '_eeg' + str(trial + 1)]
        # tmp_trial_signal = data[name + '_eeg' + str(trial + 1)]
        num_sample = int(len(tmp_trial_signal[0]) / 100)
        # num_sample = int(len(tmp_trial_signal[0]) / 200)
        print('{}-{}'.format(trial + 1, num_sample))
        label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

        temp_de = np.empty([0, num_sample])

        for channel in range(62):
            trial_signal = tmp_trial_signal[:,:,channel]

            sub0 = trial_signal[0, :]
            sub1 = trial_signal[1, :]
            sub2 = trial_signal[2, :]
            sub3 = trial_signal[3, :]
            sub4 = trial_signal[4, :]
            sub5 = trial_signal[5, :]
            sub6 = trial_signal[6, :]
            sub7 = trial_signal[7, :]

            DE_sub0 = np.zeros(shape=[0], dtype=float)
            DE_sub1 = np.zeros(shape=[0], dtype=float)
            DE_sub2 = np.zeros(shape=[0], dtype=float)
            DE_sub3 = np.zeros(shape=[0], dtype=float)
            DE_sub4 = np.zeros(shape=[0], dtype=float)
            DE_sub5 = np.zeros(shape=[0], dtype=float)
            DE_sub6 = np.zeros(shape=[0], dtype=float)
            DE_sub7 = np.zeros(shape=[0], dtype=float)

            for index in range(num_sample):
                DE_sub0 = np.append(DE_sub0, compute_DE(sub0[index * 100:(index + 1) * 100]))
                DE_sub1 = np.append(DE_sub1, compute_DE(sub1[index * 100:(index + 1) * 100]))
                DE_sub2 = np.append(DE_sub2, compute_DE(sub2[index * 100:(index + 1) * 100]))
                DE_sub3 = np.append(DE_sub3, compute_DE(sub3[index * 100:(index + 1) * 100]))
                DE_sub4 = np.append(DE_sub4, compute_DE(sub4[index * 100:(index + 1) * 100]))
                DE_sub5 = np.append(DE_sub5, compute_DE(sub5[index * 100:(index + 1) * 100]))
                DE_sub6 = np.append(DE_sub6, compute_DE(sub6[index * 100:(index + 1) * 100]))
                DE_sub7 = np.append(DE_sub7, compute_DE(sub7[index * 100:(index + 1) * 100]))

            DE_sub0_smooth = smooth(DE_sub0,5)
            DE_sub1_smooth = smooth(DE_sub1,5)
            DE_sub2_smooth = smooth(DE_sub2,5)
            DE_sub3_smooth = smooth(DE_sub3,5)
            DE_sub4_smooth = smooth(DE_sub4,5)
            DE_sub5_smooth = smooth(DE_sub5,5)
            DE_sub6_smooth = smooth(DE_sub6,5)
            DE_sub7_smooth = smooth(DE_sub7,5)

            temp_de = np.vstack([temp_de, DE_sub0_smooth])
            temp_de = np.vstack([temp_de, DE_sub1_smooth])
            temp_de = np.vstack([temp_de, DE_sub2_smooth])
            temp_de = np.vstack([temp_de, DE_sub3_smooth])
            temp_de = np.vstack([temp_de, DE_sub4_smooth])
            temp_de = np.vstack([temp_de, DE_sub5_smooth])
            temp_de = np.vstack([temp_de, DE_sub6_smooth])
            temp_de = np.vstack([temp_de, DE_sub7_smooth])

        temp_trial_de = temp_de.reshape(-1, 8, num_sample)
        temp_trial_de = temp_trial_de.transpose([2, 0, 1])
        # print(temp_trial_de.shape)


        trial_label = np.ones((num_sample, 1, 8))
        trial_label = trial_label * label[trial]
        # print(trial_label)

        temp_trial_de_new = np.append(temp_trial_de, trial_label, axis=1)
        # print('&&&&&&&&&&&&&&&&', temp_trial_de_new.shape)  #[460,63,8]

        decomposed_de = np.vstack([decomposed_de, temp_trial_de_new])  # decomposed_de:[6692,62,8] label:[6692]

    eeg_data['eeg_label'] = decomposed_de

    return eeg_data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def smooth(a,WSZ):
  # a:原始数据，NumPy 1-D array containing the data to be smoothed
  # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
  # WSZ: smoothing window size needs, which must be odd number,
  # as in the original MATLAB implementation
  out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
  r = np.arange(1,WSZ-1,2)
  start = np.cumsum(a[:WSZ-1])[::2]/r
  stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
  return np.concatenate(( start , out0, stop ))


def compute_DE(signal): # 微分熵
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def compute_VAR(signal): # 方差
    variance = np.var(signal, ddof=1)
    return variance

def compute_STD(signal):  # 标准差
    std = np.std(signal)
    return std

def compute_MAX(signal):  # 最大值
    mmax = np.max(signal)
    return mmax

def compute_MIN(signal):  # 最小值
    mmin = np.min(signal)
    return mmin

def compute_PSD(signal):  # 功率谱密度
    ft = fft(signal,512)
    abs_ft= np.abs(ft)
    abs_ft = abs_ft[:np.size(ft)//2]
    abs_ft_squa = abs_ft**2
    e = np.sum(abs_ft_squa) / np.size(abs_ft_squa)
    return e

def compute_MAV(signal):   # 平均绝对值
    abs_sig = np.abs(signal)
    abs_e = np.log2((1/np.size(signal))*(np.sum(abs_sig)))
    return abs_e

def compute_MEAN(signal):   # 平均值
    mean_sig = np.mean(signal)
    return mean_sig

# 偏度用来度量分布是否对称。正态分布左右是对称的，偏度系数为0。较大的正值表明该分布具有右侧较长尾部。较大的负值表明有左侧较长尾部
def skew_X(X):
    skewness = skew(X)
    return skewness

# 峰度系数（Kurtosis）用来度量数据在中心聚集程度。在正态分布情况下，峰度系数值是3。
# >3的峰度系数说明观察量更集中，有比正态分布更短的尾部；
# <3的峰度系数说明观测量不那么集中，有比正态分布更长的尾部，类似于矩形的均匀分布。
def kurs_X(X):
    kurs = kurtosis(X)
    return kurs

def compute_PEAK(signal):
    max_X=np.max(signal)
    min_X = np.min(signal)
    Peak = np.max([np.abs(max_X), np.abs(min_X)])
    return Peak

# 计算Petrosian's Fractal Dimension分形维数值
def Petrosian_FD(X):
    D = np.diff(X)

    delta = 0;
    N = len(X)
    # number of sign changes in signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            delta += 1

    feature = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * delta)))

    return feature

def ApEn(U, m, r):  #ApEn

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m+1) - _phi(m))


# 究极整合版
import os
import numpy as np

file_path = 'G:\Paper-EEG Emotion\SEED-tqwt2-8band-q1r3j7-2\\'
save_path = 'G:\Paper-EEG Emotion\SEED-tqwt2-8band-q1r3j7-2\Analysis-Trditional\DE0.5s-day1\\'


people_name = ['1_20131027', # 1
               '2_20140404',
               '3_20140603',
               '4_20140621',
               '5_20140411',
               '6_20130712',
               '7_20131027',
               '8_20140511',
               '9_20140620',
               '10_20131130',
               '11_20140618',
               '12_20131127',
               '13_20140527',
               '14_20140601',
               '15_20130709',]


short_name = ['djc','jl','jj',
              'lqj','ly','mhw',
              'phl', 'sxy','wk',
              'ww','wsf','wyw',
              'xyl','ys','zjy'
               ]



for i in range(len(people_name)):
    file_name = file_path + people_name[i]
    print('processing {}'.format(people_name[i]))
    decomposed_de = decompose(file_name, short_name[i])
    sio.savemat(save_path+people_name[i]+'.mat',decomposed_de)
