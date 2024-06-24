import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import random

def generate_signal(frequencies, sample_rate, duration):
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += np.sin(2.0 * np.pi * freq * t)
    return t, signal

def detect_frequencies(signal, sample_rate, target_frequencies, threshold=0.1):
    # Apply Hamming window
    window = windows.hamming(len(signal))
    signal_windowed = signal * window

    # Perform FFT
    N = len(signal)
    yf = fft(signal_windowed)
    xf = fftfreq(N, 1 / sample_rate)

    # Calculate magnitudes
    magnitudes = np.abs(yf)

    # Find indices of target frequencies
    target_indices = [np.argmin(np.abs(xf - freq)) for freq in target_frequencies]

    # Get magnitudes of target frequencies
    detected_magnitudes = {freq: magnitudes[idx] for freq, idx in zip(target_frequencies, target_indices)}

    # Determine if frequencies are detected based on the threshold
    detected = {freq: magnitude for freq, magnitude in detected_magnitudes.items() if magnitude > threshold}

    return detected

def test_detection_limit(max_frequencies, sample_rate, duration, threshold):
    detection_results = {}

    random_frequencies = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5]
    non_existing_frequencies = [1.25, 2.75, 3.75, 4.25, 5.25, 6]

    t, signal = generate_signal(random_frequencies, sample_rate, duration)
    detection_frequencies = random_frequencies + non_existing_frequencies
    detected = detect_frequencies(signal, sample_rate, detection_frequencies, threshold)

    actual_detected = [freq for freq in random_frequencies if freq in detected]
    non_existent_detected = [freq for freq in non_existing_frequencies if freq in detected]

    detection_results[0] = {
        'actual_detected': len(actual_detected),
        'non_existent_detected': len(non_existent_detected)
    }

    return detection_results

# Parameters
sample_rate = 1000  # 1000 samples per second
duration = 1.0  # 1 second duration
max_frequencies = 2000  # Maximum number of frequencies to test
threshold = 100 # Threshold for detection

# Test detection limit
detection_results = test_detection_limit(max_frequencies, sample_rate, duration, threshold)

# Print results
for num_frequencies, results in detection_results.items():
    print(f"Number of frequencies: {num_frequencies}, Actual detected: {results['actual_detected']}, Non-existent detected: {results['non_existent_detected']}")
