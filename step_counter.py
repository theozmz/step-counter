import numpy as np
import pandas as pd
from collections import deque
from scipy import interpolate
from scipy.signal import butter, sosfiltfilt, find_peaks, correlate
from scipy.signal import savgol_filter
from hampel import hampel
import matplotlib.pyplot as plt
import requests
import time
import json
import tensorflow as tf
import argparse
import sys
import os
import glob



class StepCounter:
    """
    One step counter class for both offline and real-time usage.
    You can add any other attributes you need to the class. But you should not change the interface of the class.
    """

    def __init__(self, window_sec=2.0, threshold_factor=0.8, refractory_sec=0.75, filter_len=5, 
                 cutoff_low=0.5, cutoff_high=2.0, fs=50.0, order=3, n_sigma=3.0, polyorder=3,
                 gyro_weight=0.3, noise_threshold=0.2, threshold_absolute=0.6, g=9.8,
                 gyro_energy_min=0.05, mag_var_min=0.05, penalty=0.5, acf_dominance=0.7,
                 peak_prominence=0.1, valley_prominence=0.05, min_step_period=0.3,
                 max_step_period=3.0, state_timeout=1.5, peak_valley_min_dist=0.2,
                 model_path=None,
                 gravity=False, CMA=True, hampel=True, savgol=True, LPF=True, ACF=True, FSM=False):
        """
        Initialize the step counter.
        """
        self.window_sec = window_sec
        self.threshold_factor = threshold_factor
        self.refractory_sec = refractory_sec
        self.filter_len = filter_len
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.fs = fs
        self.order = order
        self.n_sigma = n_sigma
        self.polyorder = polyorder
        self.gyro_weight = gyro_weight
        self.noise_threshold = noise_threshold
        self.threshold_absolute = threshold_absolute
        self.gyro_energy_min = gyro_energy_min
        self.mag_var_min = mag_var_min
        self.penalty = penalty
        self.acf_dominance = acf_dominance
        self.g = g
        self.gravity = gravity
        self.CMA = CMA
        self.hampel = hampel
        self.savgol = savgol
        self.LPF = LPF
        self.ACF = ACF
        # FSM
        self.FSM = FSM
        self.peak_prominence = peak_prominence
        self.valley_prominence = valley_prominence
        self.min_step_period = min_step_period
        self.max_step_period = max_step_period
        self.state_timeout = state_timeout
        self.peak_valley_min_dist = peak_valley_min_dist
        self.model = None
        if model_path is not None:
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'loss': tf.keras.losses.MeanSquaredError()
            }
            try:
                self.model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                self.model = None
        self.reset()
        # raise NotImplementedError

    def reset(self) -> None:
        """
        Reset internal state such as buffers and cumulative count.
        After reset(), total_steps should be 0.
        """
        self.total_steps = 0
        self.step_timestamps = []
        # self.time_buffer = deque()
        # self.acc_buffer = deque()
        # self.gyro_buffer = deque()
        self.time_buffer = []
        self.acc_buffer = []
        self.gyro_buffer = []
        self.mag_buffer = []
        self.last_peak_time = -np.inf
        # FSM
        # self.fsm_state = 'ZC_TO_PEAK'
        self.fsm_state = 'WAIT_PEAK'
        self.fsm_last_peak_time = -np.inf
        self.fsm_last_valley_time = -np.inf
        self.fsm_last_zc_time = -np.inf
        self.fsm_timer = 0.0
        self.fsm_signal_buffer = []
        self.fsm_last_time = None
        # raise NotImplementedError

    def _compute_magnitude(self, x):
        """
        orientation robustness
        """
        return np.sqrt(np.sum(x**2, axis=1))

    def _rotation_matrix_from_gravity(self, gx, gy, gz):
        g_norm = np.sqrt(gx**2 + gy**2 + gz**2)
        if g_norm < 1e-6:
            return np.eye(3)
        v = np.array([gx, gy, gz]) / g_norm
        u = np.array([0, 0, 1])
        axis = np.cross(v, u)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            # parallel
            return np.eye(3) if np.dot(v, u) > 0 else np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        axis = axis / axis_norm
        angle = np.arccos(np.dot(v, u))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R

    def _filter_and_rotate(self, acc, gravity=False):
        """
        Input: acc(N, 3), m/s^2, can be with or without gravity
        Output: acc_world(N, 3), m/s^2, world coordinate system
        """
        N = len(acc)
        # filters
        # gx = self._filter(acc[:, 0], self.CMA, self.hampel, self.savgol, self.LPF)
        # gy = self._filter(acc[:, 1], self.CMA, self.hampel, self.savgol, self.LPF)
        # gz = self._filter(acc[:, 2], self.CMA, self.hampel, self.savgol, self.LPF)
        gx = self._filter(acc[:, 0], False, False, False, True)
        gy = self._filter(acc[:, 1], False, False, False, True)
        gz = self._filter(acc[:, 2], False, False, False, True)
        # rotate
        a_world = np.zeros((N, 3))
        for i in range(N):
            R = self._rotation_matrix_from_gravity(gx[i], gy[i], gz[i])
            a_world[i] = R @ acc[i]
        if self.model is None:
            a_world[:, 0] = self._filter(a_world[:, 0], self.CMA, self.hampel, self.savgol, self.LPF)
            a_world[:, 1] = self._filter(a_world[:, 1], self.CMA, self.hampel, self.savgol, self.LPF)
            a_world[:, 2] = self._filter(a_world[:, 2], self.CMA, self.hampel, self.savgol, self.LPF)
        # subtract gravity if acc has
        if gravity:
            a_world[:, 2] -= self.g
        return a_world

    def _lowpass_filter(self, data, cutoff=2.0, fs=50.0, order=3):
        # Not enough data to filter safely
        if len(data) < 15:
            return data
        sos = butter(order, cutoff, 'low', fs=fs, output='sos')
        padlen = min(len(data) - 1, int(self.fs)) 
        return sosfiltfilt(sos, data, padlen=padlen)

    def _highpass_filter(self, data, cutoff=0.5, fs=50.0, order=4):
        # Not enough data to filter safely
        if len(data) < 15:
            return data
        sos = butter(order, cutoff, 'high', fs=fs, output='sos')
        padlen = min(len(data) - 1, int(self.fs)) 
        return sosfiltfilt(sos, data, padlen=padlen)

    def _hampel(self, data, window_size=2, n_sigma=3.0):
        return hampel(data, window_size=window_size, n_sigma=n_sigma).filtered_data

    def _savgol(self, data, window_size=2, polyorder=3):
        return savgol_filter(data, 2*window_size+1, polyorder)

    def _hampel_and_savgol(self, data, window_size=2, n_sigma=3.0, polyorder=3):
        filtered = hampel(data, window_size=window_size, n_sigma=n_sigma).filtered_data
        filtered = savgol_filter(filtered, 2*window_size+1, polyorder)
        return filtered

    def _causal_moving_average(self, data, filter_len=5):
        filtered = np.zeros_like(data)
        cumsum = np.cumsum(data)
        for i in range(len(data)):
            if i < filter_len:
                filtered[i] = cumsum[i] / (i + 1)
            else:
                filtered[i] = (cumsum[i] - cumsum[i - filter_len]) / filter_len
        return filtered

    def _filter(self, data, CMA=True, hampel=True, savgol=True, LPF=True):
        if CMA:
            data = self._causal_moving_average(data, self.filter_len)
        if hampel:
            data = self._hampel(data, window_size=self.filter_len, n_sigma=self.n_sigma)
        if savgol:
            data = self._savgol(data, window_size=self.filter_len, polyorder=self.polyorder)
        # data = self._hampel_and_savgol(data, window_size=self.filter_len, n_sigma=self.n_sigma, polyorder=self.polyorder)
        if LPF:
            data = self._lowpass_filter(data, cutoff=self.cutoff_high, fs=self.fs, order=self.order)
            # data = self._lowpass_filter(data, cutoff=self.cutoff_low, fs=self.fs, order=self.order)
        return data

    def _acf_period(self, signal):
        if len(signal) < int(self.fs * 0.5) or np.std(signal) < 1e-4:
            return 0
        signal = signal - np.mean(signal)
        acf = correlate(signal, signal, mode='full')
        acf = acf[len(acf)//2:]
        # human step period
        min_lag = int(0.4 * self.fs)
        max_lag = int(2.0 * self.fs)
        actual_max_lag = min(max_lag, len(acf))
        if actual_max_lag <= min_lag:
            return 0
        valid_acf = acf[min_lag:actual_max_lag]
        prominence_thresh = max(np.max(np.abs(valid_acf)) * 0.05, 1e-5)
        peaks, _ = find_peaks(valid_acf, prominence=prominence_thresh)
        if len(peaks) > 0:
            best_peak_idx = peaks[np.argmax(valid_acf[peaks])]
            best_lag = best_peak_idx + min_lag
            return best_lag / self.fs
        return 0

    def _combine_signals(self, acc_mag, gyro_mag):
        if gyro_mag is None:
            return acc_mag
        return (1 - self.gyro_weight) * acc_mag + self.gyro_weight * gyro_mag

    def _predict_window_steps(self, w_acc):
        """
        Predict step count in the window using LSTM model.
        
        Args:
            w_acc: Acceleration data with shape (window_samples, 3)
            
        Returns:
            int: Predicted step count
        """
        if self.model is None:
            return 0
        # Input shape: (batch_size=1, window_samples, 3)
        X = np.expand_dims(w_acc, axis=0)
        pred = self.model.predict(X, verbose=0)[0, 0]
        return int(round(pred))


#######################################################
# FSM

    def _fsm_update_timer(self, t):
        """Update FSM timer and return True if timeout."""
        if self.fsm_last_time is None:
            dt = 0.0
        else:
            dt = t - self.fsm_last_time
        self.fsm_timer += dt
        self.fsm_last_time = t
        return self.fsm_timer > self.state_timeout

    def _fsm_detect_peak(self, signal, t):
        """Detect local maximum in sliding window (signal has at least 3 points)."""
        if len(signal) < 3:
            return False
        if signal[-1] > signal[-2] and signal[-1] > signal[-3]:
            left_base = min(signal[-3], signal[-2])
            right_base = min(signal[-2], signal[-1]) if len(signal) > 3 else signal[-2]
            prominence = signal[-1] - max(left_base, right_base)
            if prominence >= self.peak_prominence:
                if t - self.fsm_last_peak_time >= self.peak_valley_min_dist:
                    self.fsm_last_peak_time = t
                    return True
        return False

    def _fsm_detect_valley(self, signal, t):
        """Detect local minimum in sliding window."""
        if len(signal) < 3:
            return False
        if signal[-1] < signal[-2] and signal[-1] < signal[-3]:
            left_base = max(signal[-3], signal[-2])
            right_base = max(signal[-2], signal[-1]) if len(signal) > 3 else signal[-2]
            prominence = max(left_base, right_base) - signal[-1]
            if prominence >= self.valley_prominence:
                if t - self.fsm_last_valley_time >= self.peak_valley_min_dist:
                    self.fsm_last_valley_time = t
                    return True
        return False

    def _fsm_detect_zero_crossing(self, prev, curr, t):
        """Detect zero crossing and return (zc_time, detected)."""
        if prev * curr < 0:
            if prev != curr:
                alpha = -prev / (curr - prev)
                zc_time = t - (1.0 / self.fs) * (1 - alpha)
            else:
                zc_time = t - 0.5 / self.fs
            if zc_time - self.fsm_last_zc_time > 0.1:
                self.fsm_last_zc_time = zc_time
                return zc_time, True
        return None, False

    def _fsm_update(self, t, x):
        """
        peak detection in an FSM style
        """
        step_detected = False
        # buffer
        if len(self.fsm_signal_buffer) == 0:
            self.fsm_signal_buffer.append(x)
            return step_detected, self.fsm_state
        self.fsm_signal_buffer.append(x)
        if len(self.fsm_signal_buffer) > 3:
            self.fsm_signal_buffer.pop(0)
            
        # update timer
        if self.fsm_last_time is None:
            dt = 0.0
        else:
            dt = t - self.fsm_last_time
        self.fsm_timer += dt
        self.fsm_last_time = t
        timeout = self.fsm_timer > self.state_timeout
        
        # peak valley detection
        if len(self.fsm_signal_buffer) >= 2:
            prev = self.fsm_signal_buffer[-2]
            curr = self.fsm_signal_buffer[-1]
            zc_time, zc_detected = self._fsm_detect_zero_crossing(prev, curr, t)
        else:
            zc_detected = False
        
        peak_detected = self._fsm_detect_peak(self.fsm_signal_buffer, t)
        valley_detected = self._fsm_detect_valley(self.fsm_signal_buffer, t)
        
        # reset
        if timeout:
            self.fsm_state = 'WAIT_PEAK'
            self.fsm_timer = 0.0
            return step_detected, self.fsm_state
        
        # finite state machine
        if self.fsm_state == 'WAIT_PEAK':
            if peak_detected:
                self.fsm_state = 'WAIT_ZC'
                self.fsm_timer = 0.0
        elif self.fsm_state == 'WAIT_ZC':
            if zc_detected:
                self.fsm_state = 'WAIT_VALLEY'
                self.fsm_timer = 0.0
        elif self.fsm_state == 'WAIT_VALLEY':
            if valley_detected:
                step_detected = True
                self.fsm_state = 'WAIT_PEAK'
                self.fsm_timer = 0.0
        # if self.fsm_state == 'WAIT_PEAK':
        #     if peak_detected:
        #         self.fsm_state = 'WAIT_VALLEY'
        #         self.fsm_timer = 0.0
        # elif self.fsm_state == 'WAIT_VALLEY':
        #     if valley_detected:
        #         step_detected = True
        #         self.fsm_state = 'WAIT_PEAK'
        #         self.fsm_timer = 0.0
        return step_detected, self.fsm_state


# DEBUG
    # def _fsm_update(self, t, x):
    #     """
    #     Process a single sample with the finite state machine.
    #     Returns (step_detected, current_state) where step_detected is bool.
    #     """
    #     step_detected = False

    #     # Initialize buffer
    #     if len(self.fsm_signal_buffer) == 0:
    #         self.fsm_signal_buffer.append(x)
    #         return step_detected, self.fsm_state

    #     # Append new value, keep only last 3 points
    #     self.fsm_signal_buffer.append(x)
    #     if len(self.fsm_signal_buffer) > 3:
    #         self.fsm_signal_buffer.pop(0)

    #     # Timeout and event detection
    #     timeout = self._fsm_update_timer(t)
    #     prev = self.fsm_signal_buffer[-2] if len(self.fsm_signal_buffer) >= 2 else 0
    #     curr = x
    #     _, zc_detected = self._fsm_detect_zero_crossing(prev, curr, t)
    #     peak_detected = self._fsm_detect_peak(self.fsm_signal_buffer, t)
    #     valley_detected = self._fsm_detect_valley(self.fsm_signal_buffer, t)

    #     # Abnormal interval detection (reset if too long between peaks or valleys)
    #     abnormal_interval = False
    #     if self.fsm_last_peak_time > -np.inf and t - self.fsm_last_peak_time > self.max_step_period:
    #         abnormal_interval = True
    #     if self.fsm_last_valley_time > -np.inf and t - self.fsm_last_valley_time > self.max_step_period:
    #         abnormal_interval = True

    #     # State machine transitions
        
    #     if self.fsm_state == 'WAIT_PEAK':
    #         if peak_detected:
    #             self.fsm_state = 'WAIT_VALLEY'
    #     elif self.fsm_state == 'WAIT_VALLEY':
    #         if valley_detected:
    #             step_detected = True
    #             self.fsm_state = 'WAIT_PEAK'
        
        
    #     # if self.fsm_state == 'ZC_TO_PEAK':
    #     #     if peak_detected:
    #     #         self.fsm_state = 'PEAK'
    #     #         self.fsm_timer = 0.0
    #     #     elif timeout or abnormal_interval:
    #     #         self.fsm_state = 'ZC_TO_PEAK'
    #     #         self.fsm_timer = 0.0
    #     # elif self.fsm_state == 'PEAK':
    #     #     if zc_detected:
    #     #         self.fsm_state = 'PEAK_TO_ZC'
    #     #         self.fsm_timer = 0.0
    #     #     elif timeout or abnormal_interval:
    #     #         self.fsm_state = 'ZC_TO_PEAK'
    #     #         self.fsm_timer = 0.0
    #     # elif self.fsm_state == 'PEAK_TO_ZC':
    #     #     if zc_detected:
    #     #         self.fsm_state = 'ZC_TO_VALLEY'
    #     #         self.fsm_timer = 0.0
    #     #     elif timeout or abnormal_interval:
    #     #         self.fsm_state = 'ZC_TO_PEAK'
    #     #         self.fsm_timer = 0.0
    #     # elif self.fsm_state == 'ZC_TO_VALLEY':
    #     #     if valley_detected:
    #     #         self.fsm_state = 'VALLEY'
    #     #         self.fsm_timer = 0.0
    #     #     elif timeout or abnormal_interval:
    #     #         self.fsm_state = 'ZC_TO_PEAK'
    #     #         self.fsm_timer = 0.0
    #     # elif self.fsm_state == 'VALLEY':
    #     #     if zc_detected:
    #     #         self.fsm_state = 'VALLEY_TO_ZC'
    #     #         self.fsm_timer = 0.0
    #     #     elif timeout or abnormal_interval:
    #     #         self.fsm_state = 'ZC_TO_PEAK'
    #     #         self.fsm_timer = 0.0
    #     # elif self.fsm_state == 'VALLEY_TO_ZC':
    #     #     if zc_detected:
    #     #         step_detected = True
    #     #         self.fsm_state = 'ZC_TO_PEAK'
    #     #         self.fsm_timer = 0.0
    #     #     elif timeout or abnormal_interval:
    #     #         self.fsm_state = 'ZC_TO_PEAK'
    #     #         self.fsm_timer = 0.0
    #     return step_detected, self.fsm_state
#######################################################

    def _process_window(self, w_t, w_acc, w_gyro=None, w_mag=None):
        new_peaks = []
        std_val = np.std(w_acc)
        mean_val = np.mean(w_acc)
        # denoise
        if std_val < self.noise_threshold:
            return [], 0
        # # IMU Fusion (Energy Gating)
        # if w_gyro is not None and len(w_gyro) > 0:
        #     g_mag = self._compute_magnitude(w_gyro)
        #     g_energy = np.mean(g_mag**2)
        #     if g_energy < 0.05:
        #         return [], 0
        dynamic_prominence = 0.5
        height_penalty = 0.0
        if w_gyro is not None and len(w_gyro) > 0:
            g_mag = self._compute_magnitude(w_gyro)
            g_energy = np.mean(g_mag**2)
            if g_energy < self.gyro_energy_min:
                dynamic_prominence += self.penalty
                height_penalty += self.penalty
        if w_mag is not None and len(w_mag) > 0:
            m_mag = self._compute_magnitude(w_mag)
            m_var = np.var(m_mag)
            if m_var < self.mag_var_min:
                dynamic_prominence += self.penalty
                height_penalty += self.penalty
        if dynamic_prominence > 1.0:
            return [], 0
        # ACF
        step_period = self._acf_period(w_acc) if self.ACF else 0
        # print(step_period)
        # Set minimum distance between steps to 70% of dominant ACF period
        distance_samples = max(int(step_period * self.fs * self.acf_dominance), 1)
        # distance_samples = max(int(step_period * self.acf_dominance), 1)
        # print(distance_samples)
        # distance_samples = None
        # adaptive threshold
        height_thresh = max(mean_val + self.threshold_factor * std_val, mean_val + self.threshold_absolute) 
        height_thresh += height_penalty
        # height_thresh = None
        peaks_idx, _ = find_peaks(
            w_acc, 
            height=height_thresh, 
            # prominence=0.5, 
            prominence=dynamic_prominence, 
            distance=distance_samples
        )
        for idx in peaks_idx:
            t_peak = w_t[idx]
            # refractory period to prevent double counting across windows
            if t_peak - self.last_peak_time > self.refractory_sec:
                new_peaks.append(t_peak)
                self.last_peak_time = t_peak
        return new_peaks, step_period

    def update(self, data_chunk: dict) -> dict:
        """
        Real-time update: process a chunk of new samples.

        Input
          data_chunk["time"] : numpy.ndarray with shape (M,) [required]
          data_chunk["acc"]  : numpy.ndarray with shape (M, 3) in m/s^2 [required]
          data_chunk["gyro"] : numpy.ndarray with shape (M, 3) in rad/s [optional]
          data_chunk["mag"]  : numpy.ndarray with shape (M, 3) in uT [optional]
          Chunks arrive sequentially.

        Output (must contain all keys)
          {
            "new_steps": int,
            "total_steps": int,
            "new_step_timestamps": np.ndarray,  # shape (K,), float seconds
            "diagnostics": dict
          }
        """
        # Append incoming chunks to buffers
        self.time_buffer.extend(data_chunk['time'])
        self.acc_buffer.extend(data_chunk['acc'])
        if 'gyro' in data_chunk and data_chunk['gyro'] is not None:
            self.gyro_buffer.extend(data_chunk['gyro'])
        if 'mag' in data_chunk and data_chunk['mag'] is not None:
            self.mag_buffer.extend(data_chunk['mag'])
        new_peaks_total = []
        window_samples = int(self.window_sec * self.fs)
        half = window_samples // 2
        period = 0
        model_steps = 0
        # Windowing
        while len(self.time_buffer) >= window_samples:
            ########################################################################################
            # pick up IMU signals
            w_t = np.array(self.time_buffer[:window_samples])
            w_acc = np.array(self.acc_buffer[:window_samples])
            w_gyro = np.array(self.gyro_buffer[:window_samples]) if self.gyro_buffer else None
            w_mag = np.array(self.mag_buffer[:window_samples]) if self.mag_buffer else None
            ########################################################################################
            # rotate and filters
            filtered = self._filter_and_rotate(w_acc, self.gravity)
            filt_mag = self._compute_magnitude(filtered)
            ########################################################################################
            if self.model is not None:
                steps = self._predict_window_steps(w_acc)
                model_steps += steps
                # no overlap
                self.time_buffer = self.time_buffer[window_samples:]
                self.acc_buffer = self.acc_buffer[window_samples:]
                continue
            ########################################################################################
            if self.FSM:
                # z-axis and auto g baseline
                # z_axis_signal = filtered[:, 2] 
                # w_fsm_signal = z_axis_signal - np.mean(z_axis_signal)
                w_fsm_signal = filt_mag - np.mean(filt_mag)
                # only shift
                # w_fsm_signal = filt_mag - np.mean(filt_mag)
                for i in range(len(w_t)):
                    step_detected, _ = self._fsm_update(w_t[i], w_fsm_signal[i])
                    if step_detected:
                        new_peaks_total.append(w_t[i])
                # no overlap
                self.time_buffer = self.time_buffer[window_samples:]
                self.acc_buffer = self.acc_buffer[window_samples:]
                # if self.gyro_buffer:
                #     self.gyro_buffer = self.gyro_buffer[window_samples:]
                # if self.mag_buffer:
                #     self.mag_buffer = self.mag_buffer[window_samples:]
                continue
            ########################################################################################
            # filtering
            # filt_mag = self._lowpass_filter(acc_mag) # only lowpass
            # acc_mag = self._compute_magnitude(w_acc) # only filters
            # filt_mag = self._filter(acc_mag)
            # peak detection
            detected_peaks, period = self._process_window(w_t, filt_mag, w_gyro, w_mag)
            new_peaks_total.extend(detected_peaks)
            ########################################################################################
            # remove from buffer (no overlap)
            # self.time_buffer = self.time_buffer[window_samples:]
            # self.acc_buffer = self.acc_buffer[window_samples:]
            # if self.gyro_buffer:
            #     self.gyro_buffer = self.gyro_buffer[window_samples:]
            # if self.mag_buffer:
            #     self.mag_buffer = self.mag_buffer[window_samples:]
            # remove from buffer (50% overlap)
            self.time_buffer = self.time_buffer[half:]
            self.acc_buffer = self.acc_buffer[half:]
            if self.gyro_buffer:
                self.gyro_buffer = self.gyro_buffer[half:]
            if self.mag_buffer:
                self.mag_buffer = self.mag_buffer[half:]
            ########################################################################################
        self.total_steps += len(new_peaks_total) if self.model is None else model_steps
        self.step_timestamps.extend(new_peaks_total)
        return {
            "new_steps": len(new_peaks_total) if self.model is None else model_steps,
            "total_steps": self.total_steps,
            "new_step_timestamps": np.array(new_peaks_total) if self.model is None else np.array([]),
            "diagnostics": {
                "buffer_size": len(self.time_buffer),
                "last_peak_time": self.last_peak_time,
                "period": period
            }
        }
        # raise NotImplementedError

    def run_offline(self, data: dict) -> dict:
        """
        Offline processing: process a full recording.

        Input
          data["time"] : numpy.ndarray with shape (N,) [required]
          data["acc"]  : numpy.ndarray with shape (N, 3) in m/s^2 [required]
          data["gyro"] : numpy.ndarray with shape (N, 3) in rad/s [optional]
          data["mag"]  : numpy.ndarray with shape (N, 3) in uT [optional]

        Output format for grading (must contain all keys)
          {
            "step_count": int,
            "step_timestamps": np.ndarray,  # shape (K,), float seconds
            "diagnostics": dict
          }

        Requirements on output:
          - "step_count" must be a Python int and must be >= 0.
          - "step_timestamps" must be a 1D NumPy array of dtype float with shape (K,).
            Each entry is a timestamp in seconds. If your algorithm does not produce
            timestamps, return an empty array with shape (0,) rather than None.
          - "diagnostics" must be a Python dict. It may be empty.
        """
        self.reset()
        t = data['time']
        acc = data['acc']
        gyro = data.get('gyro', None)
        mag = data.get('mag', None)
        # 1. Filtering globally
        acc = self._filter_and_rotate(acc, self.gravity) # rotate and filters
        global_filt_mag = self._compute_magnitude(acc)
        # acc_mag = self._compute_magnitude(acc) # only filters
        # global_filt_mag = self._filter(acc_mag)
        # global_filt_mag = self._lowpass_filter(acc_mag) # only lowpass
        period = 0
        # step_period = self._acf_period(global_filt_mag)
        # print(step_period)
        ########################################################################################
        # if self.FSM and self.model is None:
        #     # z-axis and auto g baseline
        #     # z_axis_signal = acc[:, 2] 
        #     # z_axis_signal = global_filt_mag - np.mean(global_filt_mag)
        #     # print(acc[:, 0] <0)
        #     # print(acc[:, 1] <0)
        #     # print(acc[:, 2] <0)
        #     # fsm_signal = z_axis_signal - np.mean(z_axis_signal)
        #     fsm_signal = global_filt_mag - np.mean(global_filt_mag)
        #     # print(fsm_signal<0)
        #     # print(np.sum(fsm_signal < 0) / len(fsm_signal))
        #     # only shift
        #     # fsm_signal = global_filt_mag - np.mean(global_filt_mag)
        #     for i in range(len(t)):
        #         step_detected, _ = self._fsm_update(t[i], fsm_signal[i])
        #         if step_detected:
        #             self.total_steps += 1
        #             self.step_timestamps.append(t[i])
        #     return {
        #         "step_count": self.total_steps,
        #         "step_timestamps": np.array(self.step_timestamps),
        #         "diagnostics": {
        #             "filtered": global_filt_mag,
        #             "fsm_signal": fsm_signal,
        #             "period": period
        #         }
        #     }
        ########################################################################################
        # 2. Windowing
        window_samples = int(self.window_sec * self.fs)
        half = window_samples // 2
        # no overlap
        # for i in range(0, len(t), window_samples):
        #     end_idx = i + window_samples
        #     if end_idx > len(t):
        #         break
        #     w_t = t[i:end_idx]
        #     w_filt_mag = global_filt_mag[i:end_idx]
        #     w_gyro = gyro[i:end_idx] if gyro is not None else None
        #     w_mag = mag[i:end_idx] if mag is not None else None
        #     # 3. Peak Detection
        #     detected_peaks, period = self._process_window(w_t, w_filt_mag, w_gyro, w_mag)
        #     self.total_steps += len(detected_peaks)
        #     self.step_timestamps.extend(detected_peaks)
        # 50% overlap
        start = 0
        while start + window_samples <= len(t):
            end = start + window_samples
            if end > len(t):
                break
            w_t = t[start:end]
            w_filt_mag = global_filt_mag[start:end]
            w_gyro = gyro[start:end] if gyro is not None else None
            w_mag = mag[start:end] if mag is not None else None
            ########################################################################################
            if self.model is not None:
                # shape: (window_samples, 3)
                steps = self._predict_window_steps(acc[start:end])
                self.total_steps += steps
                if steps > 0:
                    center_time = w_t[len(w_t)//2]
                    self.step_timestamps.append(center_time)
                # no overlap
                start += window_samples
                continue
            ########################################################################################
            if self.FSM:
                # Use zero‑mean magnitude as input signal
                fsm_signal = w_filt_mag - np.mean(w_filt_mag)
                for i in range(len(w_t)):
                    step_detected, _ = self._fsm_update(w_t[i], fsm_signal[i])
                    if step_detected:
                        self.total_steps += 1
                        self.step_timestamps.append(w_t[i])
                # no overlap
                start += window_samples
                continue
            ########################################################################################
            detected_peaks, period = self._process_window(w_t, w_filt_mag, w_gyro, w_mag)
            self.total_steps += len(detected_peaks)
            self.step_timestamps.extend(detected_peaks)
            start += half
            
        return {
            "step_count": self.total_steps,
            "step_timestamps": np.array(self.step_timestamps),
            "diagnostics": {
                "filtered": global_filt_mag,
                # "fsm_signal": fsm_signal,
                "period": period
            }
        }
        # raise NotImplementedError


class DataPreprocessor:
    """
    Preprocess data
    """

    def __init__(self):
        pass

    def offline(self, file_paths, sensor_types=None):
        data_raw = {}
        all_times = []
        for i, fpath in enumerate(file_paths):
            df = pd.read_csv(fpath)
            cols = df.columns.tolist()
            # IMU classification: use provided sensor_types list or infer from column names
            if sensor_types and i < len(sensor_types):
                stype = sensor_types[i]
            else:
                col_str = ' '.join(cols).lower()
                if 'linear acceleration' in col_str or 'acceleration' in col_str:
                    stype = 'acc'
                elif 'gyroscope' in col_str:
                    stype = 'gyro'
                elif 'magnetic' in col_str:
                    stype = 'mag'
                else:
                    print(f"Unknown IMU in {fpath}, skipped.")
                    continue
            time_col = [c for c in cols if 'Time' in c or 'time' in c][0]
            t = df[time_col].values.astype(float)
            x_col = [c for c in cols if ' x' in c.lower() and 'x (' in c][0]
            y_col = [c for c in cols if ' y' in c.lower() and 'y (' in c][0]
            z_col = [c for c in cols if ' z' in c.lower() and 'z (' in c][0]
            x = df[x_col].values.astype(float)
            y = df[y_col].values.astype(float)
            z = df[z_col].values.astype(float)
            acc_vals = np.column_stack((x, y, z))
            data_raw[stype] = (t, acc_vals)
            all_times.append(t)
        if not data_raw:
            return {}
        # interpolation
        t_acc = data_raw['acc'][0]
        result = {'time': t_acc}
        for stype, (t_orig, vals_orig) in data_raw.items():
            if stype == 'acc':
                result['acc'] = vals_orig
            else:
                interp_vals = np.zeros((len(t_acc), 3))
                for axis in range(3):
                    f_interp = interpolate.interp1d(t_orig, vals_orig[:, axis],
                                                    kind='linear', bounds_error=False,
                                                    fill_value='extrapolate')
                    interp_vals[:, axis] = f_interp(t_acc)
                result[stype] = interp_vals
        return result
        # union_t = np.sort(np.unique(np.concatenate(all_times)))
        # result = {'time': union_t}
        # for stype, (t_orig, vals_orig) in data_raw.items():
        #     interp_vals = np.zeros((len(union_t), 3))
        #     for axis in range(3):
        #         f_interp = interpolate.interp1d(t_orig, vals_orig[:, axis],
        #                                         kind='linear',
        #                                         bounds_error=False,
        #                                         fill_value='extrapolate')
        #         interp_vals[:, axis] = f_interp(union_t)
        #     result[stype] = interp_vals
        # return result

    def online(self, ip, port, buffer_names=None):
        base_url = f"http://{ip}:{port}"
        # get buffer names from /config
        if buffer_names is None:
            try:
                r = requests.get(f"{base_url}/config", timeout=3)
                configs = r.json()
                time_buf = None
                acc_bufs = [None, None, None]
                for buf in configs.get('buffers', []):
                    name = buf['name']
                    if 'time' in name or 'Time' in name:
                        time_buf = name
                    elif 'accX' in name or 'x' in name:
                        acc_bufs[0] = name
                    elif 'accY' in name or 'y' in name:
                        acc_bufs[1] = name
                    elif 'accZ' in name or 'z' in name:
                        acc_bufs[2] = name
                if time_buf and all(acc_bufs):
                    buffer_names = {'time': time_buf, 'acc': acc_bufs}
                else:
                    buffer_names = {'time': "acc_time", 'acc': ["accX", "accY", "accZ"]}
            except Exception as e:
                print("/config fail", e)
                return
        TIME_BUF = buffer_names['time']
        ACC_BUFS = buffer_names['acc']
        last_t = None
        while True:
            if last_t is None:
                params = {TIME_BUF: "", **{b: "" for b in ACC_BUFS}} # "/get?time&ax&ay&az"
            else:
                # Request only new data after last_t using time as reference.
                # For non-time buffers, use "threshold|referenceBuffer".
                params = {TIME_BUF: str(last_t)}
                for b in ACC_BUFS:
                    params[b] = f"{last_t}|{TIME_BUF}"
            try:
                r = requests.get(f"{base_url}/get", params=params, timeout=2.0)
                r.raise_for_status()
                j = r.json()
            except Exception as e:
                print("remote data fetch fail", e)
                time.sleep(0.5)
                continue
            tb = j["buffer"][TIME_BUF]["buffer"]
            axb = j["buffer"][ACC_BUFS[0]]["buffer"]
            ayb = j["buffer"][ACC_BUFS[1]]["buffer"]
            azb = j["buffer"][ACC_BUFS[2]]["buffer"]
            t = np.asarray(tb, dtype=float)
            if t.size > 0:
                acc = np.stack([
                    np.asarray(axb, dtype=float),
                    np.asarray(ayb, dtype=float),
                    np.asarray(azb, dtype=float)
                ], axis=1)
                last_t = float(t[-1])
                # yield data one by one
                # this is good for func calling outside the loop
                yield {'time': t, 'acc': acc}
            time.sleep(0.05)




class Analysis:
    """
    Matplotlib
    """

    @staticmethod
    def plot_steps(data, result, title="Step Detection Result"):
        t = data['time']
        acc = data['acc']
        mag = np.sqrt(np.sum(acc**2, axis=1))
        step_timestamps = result['step_timestamps']
        filtered = result['diagnostics']['filtered']
        plt.figure(figsize=(12, 6))
        plt.plot(t, mag, label='Raw Magnitude', alpha=0.5)
        plt.plot(t, filtered, label='Smoothed', linewidth=1)
        for ts in step_timestamps:
            plt.axvline(x=ts, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration Magnitude (m/s²)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def evaluate(predictions, ground_truths, digits=2):
        """
        OxWalk dataset evaluation
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        preds = np.array(predictions, dtype=float)
        trues = np.array(ground_truths, dtype=float)
        total_true = np.sum(trues)
        total_pred = np.sum(preds)
        abs_errors = np.abs(preds - trues)
        total_abs_error = np.sum(abs_errors)
        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = np.where(trues != 0, abs_errors / trues, 0)
        mean_rel_error = np.mean(rel_errors) * 100
        exact_matches = np.sum(preds == trues)
        accuracy = exact_matches / len(preds) * 100
        print("\n========== Step Counting Evaluation ==========")
        print(f"Number of files: {len(preds)}")
        print(f"Total true steps : {total_true:.0f}")
        print(f"Total pred steps : {total_pred:.0f}")
        print(f"Total absolute error : {total_abs_error:.0f}")
        print(f"Mean Absolute Error (MAE) : {mae:.{digits}f}")
        print(f"Root Mean Square Error (RMSE) : {rmse:.{digits}f}")
        print(f"Mean Relative Error (%) : {mean_rel_error:.{digits}f}")
        print(f"Exact Match Accuracy (%) : {accuracy:.{digits}f}")
        print("==============================================")
        return {
            'total_true': total_true,
            'total_pred': total_pred,
            'total_abs_error': total_abs_error,
            'mae': mae,
            'rmse': rmse,
            'mean_rel_error': mean_rel_error,
            'accuracy': accuracy,
        }


if __name__ == "__main__":
    # offline
    # pre = DataPreprocessor()
    # # data_dict = pre.offline(["data/acc_test.csv", "data/gyro_test.csv"])
    # # data_dict = pre.offline(["data/acc_test.csv"])
    # # data_dict = pre.offline(["data/acc32.csv"])
    # data_dict = pre.offline(["data/acc60.csv"])
    # # data_dict = pre.offline(["data/acc60.csv", "data/gyro60.csv", "data/mag60.csv"])
    # # data_dict = pre.offline(["data/acc63.csv"])
    # counter = StepCounter()
    # result = counter.run_offline(data_dict)
    # print("step:", result['step_count'])
    # Analysis.plot_steps(data_dict, result)
    
    # model offline
    # pre = DataPreprocessor()
    # # data_dict = pre.offline(["data/acc_test.csv", "data/gyro_test.csv"])
    # # data_dict = pre.offline(["data/acc_test.csv"])
    # # data_dict = pre.offline(["data/acc32.csv"])
    # data_dict = pre.offline(["data/acc60.csv"])
    # # data_dict = pre.offline(["data/acc60.csv", "data/gyro60.csv", "data/mag60.csv"])
    # # data_dict = pre.offline(["data/acc63.csv"])
    # counter = StepCounter(model_path='step_counter_lstm.h5')
    # result = counter.run_offline(data_dict)
    # print("step:", result['step_count'])
    # # Analysis.plot_steps(data_dict, result)
    
    # online
    # pre = DataPreprocessor()
    # gen = pre.online("192.168.0.247", 8080)
    # counter = StepCounter()
    # for chunk in gen:
    #     out = counter.update(chunk)
    #     if out['new_steps'] > 0:
    #         print("new:", out['new_steps'], "total:", out['total_steps'])
    # pass
    
    
    parser = argparse.ArgumentParser(description="Step Counter with evaluation")
    parser.add_argument('--mode', choices=['offline', 'online'], required=True, help="Processing mode: offline (evaluate on files) or online (real-time)")
    parser.add_argument('--files', type=str, nargs='+', default=None, help="List of file paths for simple offline testing (optional)")
    parser.add_argument('--data_dir', type=str, default='.', help="Directory containing processed CSV files (offline mode only)")
    parser.add_argument('--ip', type=str, default='127.0.0.1', help="IP address for online mode")
    parser.add_argument('--port', type=int, default=8080, help="Port for online mode")
    parser.add_argument('--model_path', type=str, default=None, help="Path to trained LSTM model (optional)")
    parser.add_argument('--window_sec', type=float, default=2.0)
    parser.add_argument('--threshold_factor', type=float, default=0.8)
    parser.add_argument('--refractory_sec', type=float, default=0.75)
    parser.add_argument('--filter_len', type=int, default=5)
    parser.add_argument('--cutoff_low', type=float, default=0.5)
    parser.add_argument('--cutoff_high', type=float, default=2.0)
    parser.add_argument('--fs', type=float, default=50.0)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--n_sigma', type=float, default=3.0)
    parser.add_argument('--polyorder', type=int, default=3)
    parser.add_argument('--gyro_weight', type=float, default=0.3)
    parser.add_argument('--noise_threshold', type=float, default=0.2)
    parser.add_argument('--threshold_absolute', type=float, default=0.6)
    parser.add_argument('--gyro_energy_min', type=float, default=0.05)
    parser.add_argument('--mag_var_min', type=float, default=0.05)
    parser.add_argument('--penalty', type=float, default=0.5)
    parser.add_argument('--acf_dominance', type=float, default=0.7)
    parser.add_argument('--g', type=float, default=9.8)
    parser.add_argument('--gravity', type=bool, default=False)
    parser.add_argument('--CMA', type=bool, default=True)
    parser.add_argument('--hampel', type=bool, default=True)
    parser.add_argument('--savgol', type=bool, default=True)
    parser.add_argument('--LPF', type=bool, default=True)
    parser.add_argument('--ACF', type=bool, default=True)
    # gravity=False, CMA=True, hampel=True, savgol=True, LPF=True, ACF=True
    parser.add_argument('--FSM', type=bool, default=False)
    parser.add_argument('--peak_prominence', type=float, default=0.1)
    parser.add_argument('--valley_prominence', type=float, default=0.05)
    parser.add_argument('--min_step_period', type=float, default=0.3)
    parser.add_argument('--max_step_period', type=float, default=3.0)
    parser.add_argument('--state_timeout', type=float, default=1.5)
    parser.add_argument('--peak_valley_min_dist', type=float, default=0.2)
    args = parser.parse_args()
    
    counter = StepCounter(
        window_sec=args.window_sec,
        threshold_factor=args.threshold_factor,
        refractory_sec=args.refractory_sec,
        filter_len=args.filter_len,
        cutoff_low=args.cutoff_low,
        cutoff_high=args.cutoff_high,
        fs=args.fs,
        order=args.order,
        n_sigma=args.n_sigma,
        polyorder=args.polyorder,
        gyro_weight=args.gyro_weight,
        noise_threshold=args.noise_threshold,
        threshold_absolute=args.threshold_absolute,
        gyro_energy_min=args.gyro_energy_min,
        mag_var_min=args.mag_var_min,
        penalty=args.penalty,
        acf_dominance=args.acf_dominance,
        model_path=args.model_path,
        g=args.g,
        gravity=args.gravity,
        CMA=args.CMA,
        hampel=args.hampel,
        savgol=args.savgol,
        LPF=args.LPF,
        ACF=args.ACF,
        FSM=args.FSM,
        peak_prominence=args.peak_prominence,
        valley_prominence=args.valley_prominence,
        min_step_period=args.min_step_period,
        max_step_period=args.max_step_period,
        state_timeout=args.state_timeout,
        peak_valley_min_dist=args.peak_valley_min_dist
    )
    
    if args.mode == 'offline':
        # Simple test mode with specified files
        if args.files:
            print(f"Running simple offline test with {len(args.files)} file(s)...")
            preprocessor = DataPreprocessor()
            try:
                data_dict = preprocessor.offline(args.files)
                if not data_dict:
                    print("Warning: Could not read files.")
                    sys.exit(1)
                result = counter.run_offline(data_dict)
                print(f"\n========== Step Counting Result ==========")
                print(f"Total steps detected: {result['step_count']}")
                print(f"Number of step timestamps: {len(result['step_timestamps'])}")
                if result['step_timestamps'].size > 0:
                    print(f"First step timestamp: {result['step_timestamps'][0]:.2f}s")
                    print(f"Last step timestamp: {result['step_timestamps'][-1]:.2f}s")
                print("==============================================")
            except Exception as e:
                print(f"Error processing files: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        # Evaluation mode with directory
        elif not os.path.isdir(args.data_dir):
            print(f"Error: Data directory '{args.data_dir}' does not exist.")
            sys.exit(1)
        else:
            file_pattern = os.path.join(args.data_dir, "*.csv")
            all_files = glob.glob(file_pattern)
            if not all_files:
                print(f"No CSV files found in '{args.data_dir}'")
                sys.exit(1)
            valid_files = []
            true_steps_list = []
            for f in all_files:
                basename = os.path.basename(f)
                parts = basename.split('_')
                if len(parts) != 2 or not parts[1].endswith('.csv'):
                    print(f"Skipping file with unexpected name: {basename}")
                    continue
                try:
                    true_steps = int(parts[1].replace('.csv', ''))
                except ValueError:
                    print(f"Skipping file with invalid true steps: {basename}")
                    continue
                valid_files.append(f)
                true_steps_list.append(true_steps)
            if not valid_files:
                print("No valid files to process.")
                sys.exit(1)
            print(f"Found {len(valid_files)} valid files.")
            preprocessor = DataPreprocessor()
            predictions = []
            
            for f, true_steps in zip(valid_files, true_steps_list):
                try:
                    data_dict = preprocessor.offline([f])
                    if not data_dict:
                        print(f"Warning: Could not read file {f}, skipping.")
                        continue
                    result = counter.run_offline(data_dict)
                    pred_steps = result['step_count']
                    predictions.append(pred_steps)
                    print(f"File: {os.path.basename(f)} -> True: {true_steps}, Pred: {pred_steps}")
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    continue
                
            if predictions:
                Analysis.evaluate(predictions, true_steps_list)
            else:
                print("No predictions were made.")
                
    elif args.mode == 'online':
        print("Starting online mode (real-time step counting).")
        print("Press Ctrl+C to stop.")
        preprocessor = DataPreprocessor()
        gen = preprocessor.online(args.ip, args.port)
        try:
            for chunk in gen:
                out = counter.update(chunk)
                if out['new_steps'] > 0:
                    print(f"New steps: {out['new_steps']}, Total: {out['total_steps']}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
        except Exception as e:
            print(f"Error: {e}")
