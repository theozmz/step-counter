# Step Counter - IMU-based Step Detection System

[[English](README.md) | [中文](README-zh.md)]

> The most profound technologies are those that disappear. They weave themselves into the fabric of everyday life until they are indistinguishable from it. -- Mark Weiser (The Father of Ubiquitous Computing)

**SOURCE CODE WILL NOT BE RELEASED UNTIL GRADED.**

## Overview

A comprehensive step counting system based on Inertial Measurement Unit (IMU) sensor data. This project provides both offline batch processing and real-time online step detection capabilities with configurable signal processing algorithms and optional LSTM deep learning model support.

## Features

- **Dual Mode Operation**: Support for both offline CSV file processing and real-time online data streaming
- **Advanced Signal Processing**: Multiple filtering and analysis algorithms including:
  - Butterworth Low-Pass Filter (LPF)
  - Hampel filter for outlier removal
  - Savitzky-Golay filter for smoothing
  - Center of Mass Acceleration (CMA) calculation
  - Autocorrelation Function (ACF) analysis
  - Finite State Machine (FSM) for step detection
- **Deep Learning Integration**: Optional LSTM model for enhanced step detection accuracy
- **Configurable Parameters**: Extensive parameter tuning for different use cases and sensor configurations
- **Ablation Study Support**: Built-in evaluation framework for algorithm component analysis
- **GUI Interface**: PyQt5-based graphical user interface for easy configuration and visualization
- **Real-time Visualization**: Live plotting of sensor data and detected steps

## Project Structure

```
github/
├── demo.py              # Main GUI application with offline, online, and evaluation modes
├── step_counter.py      # Core step counter implementation
├── train.py             # LSTM model training script
├── README.md            # English documentation (this file)
└── README-zh.md         # Chinese documentation
```

## Requirements

### Python Dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn tensorflow PyQt5 hampel requests
```
or
```bash
pip install -r requirements.txt
```

### System Requirements

- Python 3.7 or higher
- PyQt5 for GUI functionality
- TensorFlow/Keras for LSTM model support (optional)

## Usage

### 1. Offline Mode

Process CSV files containing IMU sensor data:

```bash
python demo.py
```

Then select "Offline Step Counter" from the GUI and:
1. Browse and select your CSV files
2. Configure parameters as needed
3. Click OK to start processing

### 2. Online Mode

Connect to a real-time data stream:

```bash
python demo.py
```

Then select "Online Step Counter" from the GUI and:
1. Enter IP address and port of the data source
2. Configure parameters
3. Click OK to start real-time detection

### 3. Training LSTM Model

Train a custom LSTM model for step detection:

```bash
python train.py
```

Configure the training parameters in `train.py` before running:
- `INPUT_ROOT`: Path to raw training data
- `OUTPUT_ROOT`: Directory for saving trained models
- `MODEL_NAME`: Name of the output model file

### 4. Ablation Study

Evaluate different algorithm configurations:

```bash
python demo.py
```

Select "Evaluation" mode to run ablation studies with multiple parameter combinations.

## Configuration Parameters

### Signal Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_sec` | 2.0 | Analysis window size in seconds |
| `threshold_factor` | 0.8 | Threshold multiplier for step detection |
| `refractory_sec` | 0.75 | Minimum time between consecutive steps |
| `filter_len` | 5 | Filter length for smoothing |
| `cutoff_low` | 0.5 | Low cutoff frequency for bandpass filter (Hz) |
| `cutoff_high` | 2.0 | High cutoff frequency for bandpass filter (Hz) |
| `fs` | 50.0 | Sampling frequency (Hz) |
| `order` | 3 | Filter order |
| `n_sigma` | 3.0 | Number of standard deviations for outlier detection |
| `polyorder` | 3 | Polynomial order for Savitzky-Golay filter |

### Algorithm Flags

- **gravity**: Enable gravity compensation
- **CMA**: Enable Center of Mass Acceleration calculation
- **hampel**: Enable Hampel filter for outlier removal
- **savgol**: Enable Savitzky-Golay smoothing filter
- **LPF**: Enable Low-Pass Filter
- **ACF**: Enable Autocorrelation Function analysis
- **FSM**: Enable Finite State Machine for step detection

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gyro_weight` | 0.3 | Weight for gyroscope data fusion |
| `noise_threshold` | 0.2 | Noise threshold for signal quality assessment |
| `threshold_absolute` | 0.6 | Absolute threshold for step detection |
| `gyro_energy_min` | 0.05 | Minimum gyroscope energy threshold |
| `mag_var_min` | 0.05 | Minimum magnetometer variance threshold |
| `penalty` | 0.5 | Penalty factor for false positive reduction |
| `acf_dominance` | 0.7 | ACF dominance threshold |

### FSM Parameters (when FSM enabled)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `peak_prominence` | 0.1 | Minimum peak prominence |
| `valley_prominence` | 0.05 | Minimum valley prominence |
| `min_step_period` | 0.3 | Minimum step period (seconds) |
| `max_step_period` | 3.0 | Maximum step period (seconds) |
| `state_timeout` | 1.5 | State timeout duration (seconds) |
| `peak_valley_min_dist` | 0.2 | Minimum distance between peak and valley |

## Input Data Format

CSV files should contain the following columns:
- Timestamp information
- Accelerometer data (ax, ay, az)
- Gyroscope data (gx, gy, gz)
- Magnetometer data (mx, my, mz) - optional

Example format:
```csv
timestamp,ax,ay,az,gx,gy,gz,mx,my,mz
2024-01-01 10:00:00.000,0.1,0.2,9.8,0.01,0.02,0.03,0.1,0.2,0.3
...
```

## Output

The system provides:
- Total step count
- Step timestamps
- Visual plots of sensor data with detected steps marked
- Real-time step count updates (online mode)
- Evaluation metrics (ablation study mode)

## Architecture

### Core Components

1. **StepCounter Class** (`step_counter.py`)
   - Main step detection engine
   - Implements all signal processing algorithms
   - Supports both batch and streaming modes

2. **DataPreprocessor Class** (`step_counter.py`)
   - Data loading and preprocessing
   - Sensor data normalization
   - Time series interpolation

3. **Analysis Class** (`step_counter.py`)
   - Performance evaluation and metrics
   - Result visualization
   - Statistical analysis

4. **GUI Application** (`demo.py`)
   - PyQt5-based user interface
   - Three operational modes: Offline, Online, Evaluation
   - Real-time data visualization

5. **Training Pipeline** (`train.py`)
   - LSTM model training
   - Data preparation and augmentation
   - Model evaluation and export

## Customization

### Adding New Algorithms

To add new signal processing algorithms:
1. Implement the algorithm in `StepCounter` class
2. Add corresponding parameters to the parameter dialogs
3. Update the processing pipeline in `step_counter.py`

### Modifying LSTM Model

Edit `train.py` to customize the LSTM architecture:
- Change number of LSTM units
- Modify network depth
- Adjust training hyperparameters

## Troubleshooting

### Common Issues

1. **PyQt5 Import Error**
   ```bash
   pip install PyQt5
   ```

2. **TensorFlow Not Found** (if using LSTM)
   ```bash
   pip install tensorflow
   ```

3. **CSV File Format Error**
   - Ensure CSV files have correct column headers
   - Check timestamp format consistency
   - Verify numeric data types for sensor readings

4. **Connection Refused** (Online Mode)
   - Verify IP address and port settings
   - Ensure data server is running
   - Check firewall settings

## Performance Tips

- Use appropriate `window_sec` for your application (shorter for real-time, longer for accuracy)
- Enable only necessary algorithm flags to reduce computational overhead
- Adjust `threshold_factor` based on sensor noise characteristics
- For noisy environments, increase `n_sigma` and enable all filters
- Use LSTM model only when traditional methods are insufficient

## License

This project is for educational and research purposes.

## Acknowledgments

- HKU Course 7310
- Contributors and maintainers

## Contact

For questions or issues, please refer to course materials or contact the project maintainer.
