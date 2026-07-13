import os
from datetime import datetime
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, resample, decimate
from scipy.spatial.transform import Rotation as R_scipy
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Masking
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


G = 9.80665
CUTOFF_HIGH = 0.5 # HPF cutoff
CUTOFF_GRAVITY = 1.0
ORIG_FS = 100.0 # OxWalk fs
TARGET_FS = 50.0 # step_counter fs
WINDOW_LEN = 2.0
STEP_LEN = 1.0
LSTM_UNITS = 64
DROPOUT = 0.5
EPOCHS = 30
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42


FOLDERS = ['Hip_100Hz', 'Wrist_100Hz']
INPUT_ROOT = 'D:\\code\\hku\\7310\\individual\\data\\raw'
OUTPUT_ROOT = 'd5'
MODEL_NAME = 'm5.h5'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

metadata = []   # (idx, participant_id, total_steps, original_file)


def parse_timestamp_series(ts_series):
    first_ts = ts_series.iloc[0] if isinstance(ts_series, pd.Series) else ts_series[0]
    if '-' in first_ts or ' ' in first_ts:
        try:
            dt_list = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') for ts in ts_series]
        except ValueError:
            dt_list = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in ts_series]
        base = dt_list[0]
        return np.array([(dt - base).total_seconds() for dt in dt_list])
    else:
        prev_minute = None
        hour = 0
        abs_seconds = []
        for ts in ts_series:
            parts = ts.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            if prev_minute is not None and minutes < prev_minute:
                hour += 1
            prev_minute = minutes
            total_sec = hour * 3600 + minutes * 60 + seconds
            abs_seconds.append(total_sec)
        return np.array(abs_seconds)

def highpass_filter(data, cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def rotation_matrix_from_gravity(gx, gy, gz):
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

def vectorized_world_frame_conversion(acc, gravity):
    """
    acc: shape (N, 3)
    gravity: shape (N, 3)
    """
    g_norm = np.linalg.norm(gravity, axis=1, keepdims=True)
    g_norm = np.where(g_norm < 1e-6, 1.0, g_norm)
    v = gravity / g_norm
    u = np.zeros_like(v)
    u[:, 2] = 1.0
    axis = np.cross(v, u)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis_norm_safe = np.where(axis_norm < 1e-6, 1.0, axis_norm)
    axis_normalized = axis / axis_norm_safe
    angle = np.arccos(np.clip(np.sum(v * u, axis=1), -1.0, 1.0))
    rotvecs = axis_normalized * angle[:, np.newaxis]
    r = R_scipy.from_rotvec(rotvecs)
    a_world = r.apply(acc)
    return a_world

def preprocess_and_save(file_path, idx):
    print(f"Processing [{idx}]: {file_path}")
    df = pd.read_csv(file_path)
    df['abs_time'] = parse_timestamp_series(df['timestamp'])
    df_resampled = resample_dataframe(df, ORIG_FS, TARGET_FS)
    x = df_resampled['x'].values * G
    y = df_resampled['y'].values * G
    z = df_resampled['z'].values * G
    # x = df_resampled['x'].values
    # y = df_resampled['y'].values
    # z = df_resampled['z'].values
    fs = TARGET_FS
    gx = lowpass_filter(x, CUTOFF_GRAVITY, fs)
    gy = lowpass_filter(y, CUTOFF_GRAVITY, fs)
    gz = lowpass_filter(z, CUTOFF_GRAVITY, fs)
    N = len(x)
    a_world = np.zeros((N, 3))
    acc_sensor = np.column_stack((x, y, z))
    gravity_components = np.column_stack((gx, gy, gz))
    a_world = vectorized_world_frame_conversion(acc_sensor, gravity_components)
    a_lin_world = a_world.copy()
    a_lin_world[:, 0] -= G
    x_lin = highpass_filter(a_lin_world[:, 0], CUTOFF_HIGH, fs)
    y_lin = highpass_filter(a_lin_world[:, 1], CUTOFF_HIGH, fs)
    z_lin = highpass_filter(a_lin_world[:, 2], CUTOFF_HIGH, fs)
    total_steps = int(df_resampled['annotation'].values.round().sum())
    out_df = pd.DataFrame({
        'Time (s)': df_resampled['time_rel'],
        'Acceleration x (m/s^2)': x_lin,
        'Acceleration y (m/s^2)': y_lin,
        'Acceleration z (m/s^2)': z_lin,
        'annotation': df_resampled['annotation']
    })
    
    out_name = f"{idx}_{total_steps}.csv"
    out_path = os.path.join(OUTPUT_ROOT, out_name)
    out_df.to_csv(out_path, index=False, float_format='%.6f')
    participant_id = os.path.basename(file_path).split('_')[0]
    metadata.append({
        'idx': idx,
        'participant_id': participant_id,
        'total_steps': total_steps,
        'original_file': file_path,
        'output_file': out_name
    })
    print(f"Saved: {out_path} (total steps: {total_steps})")
    return out_path

def resample_dataframe(df, original_fs, target_fs):
    duration = df['abs_time'].max() - df['abs_time'].min()
    n_new = int(duration * target_fs) + 1
    new_times = np.linspace(df['abs_time'].min(), df['abs_time'].max(), n_new)
    
    # resampled = {}
    # for col in ['x', 'y', 'z', 'annotation']:
    #     interp_func = np.interp(new_times, df['abs_time'], df[col].values)
    #     resampled[col] = interp_func
    
    resampled = {}
    for col in ['x', 'y', 'z']:
        resampled[col] = np.interp(new_times, df['abs_time'], df[col].values)
    interp_ann = interp1d(df['abs_time'], df['annotation'].values,
                          kind='nearest', bounds_error=False, fill_value='extrapolate')
    resampled['annotation'] = interp_ann(new_times).astype(int)
    
    resampled['time_rel'] = new_times - new_times[0]
    resampled['abs_time'] = new_times
    return pd.DataFrame(resampled)



all_csv_files = []
for folder in FOLDERS:
    folder_path = os.path.join(INPUT_ROOT, folder)
    if os.path.isdir(folder_path):
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        all_csv_files.extend(csv_files)
all_csv_files.sort()
global_idx = 0
for file in all_csv_files:
    preprocess_and_save(file, global_idx)
    global_idx += 1
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(OUTPUT_ROOT, 'metadata.csv'), index=False)
print(f"Preprocessed {len(metadata)} files. Metadata saved to {os.path.join(OUTPUT_ROOT, 'metadata.csv')}")



def build_window_dataset(processed_dir, window_len_sec, step_len_sec, fs):
    window_samples = int(window_len_sec * fs)
    step_samples = int(step_len_sec * fs)
    X_all = []
    y_all = []
    groups_all = []
    meta = pd.read_csv(os.path.join(processed_dir, 'metadata.csv'))
    
    for _, row in meta.iterrows():
        idx = row['idx']
        participant_id = row['participant_id']
        csv_file = os.path.join(processed_dir, row['output_file'])
        df = pd.read_csv(csv_file)
        acc = df[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']].values
        annot = df['annotation'].values.round().astype(int)
        
        n = len(acc)
        for start in range(0, n - window_samples + 1, step_samples):
            window = acc[start:start+window_samples]
            steps = annot[start:start+window_samples].sum()
            X_all.append(window)
            y_all.append(steps)
            groups_all.append(participant_id)
    
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    groups = np.array(groups_all)
    return X, y, groups




X, y, groups = build_window_dataset(OUTPUT_ROOT, WINDOW_LEN, STEP_LEN, TARGET_FS)
print(f"Dataset shape: X={X.shape}, y={y.shape}, Groups={len(np.unique(groups))}")
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups=groups))
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
model = Sequential([
    Bidirectional(LSTM(LSTM_UNITS, input_shape=(X.shape[1], 3), return_sequences=False)),
    Dropout(DROPOUT),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
y_pred = model.predict(X_val).flatten()
mae = mean_absolute_error(y_val, y_pred)
print(f"Validation MAE: {mae:.3f} steps/window")
model.save(MODEL_NAME)
print(f"Model saved to {MODEL_NAME}")