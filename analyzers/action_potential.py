"""
Action potential analysis module for neuronal data processing.
Provides tools for spike detection and feature extraction from voltage traces.
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import logging
from config import Config

logger = logging.getLogger(__name__)

class ActionPotentialAnalyzer:
    """Analyzes action potential waveforms and their characteristics."""
    
    def __init__(self, dataset, sampling_rate: float):
        """
        Initialize the analyzer with a dataset and sampling rate.
        
        The Allen Cell Types Database provides voltage recordings in a special format.
        We need to properly scale and offset these recordings to get meaningful values.
        
        Args:
            dataset: Dictionary containing voltage recordings and metadata
            sampling_rate: Recording sampling rate in Hz
        """
        self.dataset = dataset
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # Allen database typically needs these detection parameters
        self.threshold = 0.0  # mV (will be adjusted based on data)
        self.window = int(0.002 * sampling_rate)  # 2ms window around spikes
        
        # Analyze voltage range to set appropriate threshold
        voltage = dataset['response']
        self.baseline = np.median(voltage)
        voltage_std = np.std(voltage)
        
        # Set threshold to 3 standard deviations above baseline
        self.threshold = self.baseline + 3 * voltage_std
        
        logger.info(f"Initialized analyzer - baseline: {self.baseline:.2f} mV, "
                   f"threshold: {self.threshold:.2f} mV")
    
    def detect_spikes(self, voltage: np.ndarray, 
                     threshold: float = None,
                     min_height: float = 0.0) -> np.ndarray:
        """
        Detect action potentials using adaptive thresholding.
        
        Args:
            voltage: Membrane potential recordings (mV)
            threshold: Optional override for spike threshold
            min_height: Minimum peak prominence above baseline
        """
        try:
            if threshold is None:
                threshold = self.threshold
            
            # Calculate voltage statistics for diagnostics
            v_mean = np.mean(voltage)
            v_std = np.std(voltage)
            logger.info(f"Voltage stats - mean: {v_mean:.2f} mV, std: {v_std:.2f} mV")
            
            # Find peaks with strict criteria for real spikes
            peaks, properties = signal.find_peaks(
                voltage,
                height=threshold,
                distance=int(0.001 * self.sampling_rate),  # Minimum 1ms between spikes
                prominence=2 * v_std,  # Peaks must stand out from noise
                width=int(0.0005 * self.sampling_rate)  # Typical spike width ~0.5ms
            )
            
            if len(peaks) > 0:
                # Verify these are likely real spikes
                peak_heights = voltage[peaks] - self.baseline
                valid_peaks = peaks[peak_heights > 20.0]  # Real spikes are usually >20mV
                
                if len(valid_peaks) < len(peaks):
                    logger.warning(f"Filtered out {len(peaks) - len(valid_peaks)} "
                                 f"small events that were likely not spikes")
                    peaks = valid_peaks
            
            logger.info(f"Detected {len(peaks)} valid action potentials")
            return peaks
            
        except Exception as e:
            logger.error(f"Error in spike detection: {e}")
            raise
    
    def calculate_features(self, time: np.ndarray, waveform: np.ndarray) -> Dict[str, float]:
        """Calculate key features of an action potential waveform."""
        try:
            features = {}
            
            # Use first 20% of points before spike for baseline
            pre_spike = int(0.2 * len(waveform))
            baseline = np.median(waveform[:pre_spike])
            
            # Basic features
            peak_idx = np.argmax(waveform)
            peak_voltage = waveform[peak_idx]
            features['amplitude'] = float(peak_voltage - baseline)
            
            # Only continue if this looks like a real spike
            if features['amplitude'] < 20.0:  # Real spikes are usually >20mV
                raise ValueError("Amplitude too small for a real action potential")
            
            # Width calculation (at half maximum)
            half_height = baseline + features['amplitude'] / 2
            above_half = waveform > half_height
            crossings = np.where(np.diff(above_half.astype(int)))[0]
            
            if len(crossings) >= 2:
                features['width'] = float(time[crossings[-1]] - time[crossings[0]])
            else:
                features['width'] = np.nan
                
            # After-hyperpolarization
            post_peak = waveform[peak_idx:]
            features['ahp'] = float(np.min(post_peak) - baseline)
            
            # Maximum rise rate
            dv = np.diff(waveform)
            dt = np.diff(time)
            features['max_dvdt'] = float(np.max(dv/dt[0]))
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise
    def _calculate_phase_plot_slope(self, voltage: np.ndarray) -> float:
        # """
        # Calculate the maximum slope of the phase plot (dV/dt vs V).
        # Includes protection against divide by zero errors.
        # """
        try:
            dv = np.diff(voltage)
            dvdt = dv / self.dt
            v_phase = voltage[:-1]  # Voltage points for phase plot
            
            # Find region of maximum slope (during spike upstroke)
            max_dvdt_idx = np.argmax(dvdt)
            
            # Calculate slope in phase plot with safety checks
            if (max_dvdt_idx > 0 and max_dvdt_idx < len(dvdt)-1):
                denominator = v_phase[max_dvdt_idx+1] - v_phase[max_dvdt_idx-1]
                # Check for divide by zero
                if abs(denominator) > 1e-10:  # Small threshold to avoid division by very small numbers
                    slope = (dvdt[max_dvdt_idx+1] - dvdt[max_dvdt_idx-1]) / denominator
                    return slope
            return np.nan
            
        except Exception as e:
            logger.error(f"Error calculating phase plot slope: {e}")
            return np.nan
    def analyze_sweep(self, sweep_number: int) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        Analyze all spikes in a sweep.
        
        Args:
            sweep_number (int): Sweep number to analyze
            
        Returns:
            tuple: (spike_times, list_of_feature_dictionaries)
        """
        try:
            # Get sweep data
            sweep_data = self.dataset.get_sweep(sweep_number)
            v = sweep_data['response']
            t = np.arange(len(v)) / self.sampling_rate
            
            # Detect spikes
            peaks = self.detect_spikes(v)
            all_features = []
            
            # Analyze each spike
            for peak in peaks:
                time, waveform = self.extract_waveform(v, peak)
                features = self.calculate_features(time, waveform)
                all_features.append(features)
            
            logger.info(f"Analyzed {len(peaks)} spikes in sweep {sweep_number}")
            return t[peaks], all_features
            
        except Exception as e:
            logger.error(f"Error analyzing sweep {sweep_number}: {e}")
            raise
    
    def get_firing_rate(self, sweep_number: int) -> float:
        """
        Calculate the average firing rate for a sweep.
        
        Args:
            sweep_number (int): Sweep number to analyze
            
        Returns:
            float: Firing rate in Hz
        """
        try:
            sweep_data = self.dataset.get_sweep(sweep_number)
            v = sweep_data['response']
            duration = len(v) / self.sampling_rate
            
            peaks = self.detect_spikes(v)
            rate = len(peaks) / duration
            
            logger.debug(f"Firing rate for sweep {sweep_number}: {rate:.2f} Hz")
            return rate
            
        except Exception as e:
            logger.error(f"Error calculating firing rate for sweep {sweep_number}: {e}")
            raise