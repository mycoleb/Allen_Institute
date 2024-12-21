"""
Visualizer for neural recording data from the Allen Cell Types Database.
Creates publication-quality plots of voltage recordings and analyses.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Set up seaborn style for better-looking plots
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]

logger = logging.getLogger(__name__)

class APVisualizer:
    """Visualizes neuronal recordings and analyses."""
    
    def __init__(self, save_dir: Optional[str] = 'figures'):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save generated figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Color palette for different stimulus types
        self.stim_colors = {
            'Short Square': 'blue',
            'Short Square - Triple': 'darkblue',
            'Long Square': 'green',
            'Ramp': 'red',
            'Noise': 'purple',
            'Square - 2s Suprathreshold': 'orange'
        }
    
    def plot_sweep(self, time: np.ndarray, voltage: np.ndarray, 
                  stimulus: np.ndarray, sweep_info: Dict,
                  show_stimulus: bool = True) -> None:
        """
        Plot a single sweep of recording data.
        
        Args:
            time: Time points in seconds
            voltage: Voltage recordings in mV
            stimulus: Current stimulus in pA
            sweep_info: Dictionary with sweep metadata
            show_stimulus: Whether to show stimulus trace
        """
        if show_stimulus:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            
            # Plot voltage trace
            ax1.plot(time * 1000, voltage, 'k-', lw=1)
            ax1.set_ylabel('Membrane Potential (mV)')
            ax1.set_title(f"Sweep {sweep_info['sweep_number']}: "
                         f"{sweep_info['stimulus_type'].decode()}")
            
            # Plot stimulus
            ax2.plot(time * 1000, stimulus, 'gray', lw=1)
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (pA)')
        else:
            # Create figure with single subplot
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            
            # Plot voltage trace
            ax1.plot(time * 1000, voltage, 'k-', lw=1)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Membrane Potential (mV)')
            ax1.set_title(f"Sweep {sweep_info['sweep_number']}: "
                         f"{sweep_info['stimulus_type'].decode()}")
            
        plt.tight_layout()
    
    def plot_stimulus_comparison(self, sweep_data: List[Dict]) -> None:
        """
        Create a comparison plot of different stimulus types.
        
        Args:
            sweep_data: List of dictionaries containing sweep information
        """
        # Group sweeps by stimulus type
        stim_types = {}
        for sweep in sweep_data:
            stim_type = sweep['stimulus_type'].decode()
            if stim_type not in stim_types:
                stim_types[stim_type] = []
            stim_types[stim_type].append(sweep)
            
        # Create plot
        plt.figure(figsize=(12, 6))
        
        for i, (stim_type, sweeps) in enumerate(stim_types.items()):
            # Get voltage ranges for this stimulus type
            v_ranges = [s['voltage_range'] for s in sweeps]
            amplitudes = [s['amplitude'] for s in sweeps]
            
            # Create scatter plot
            plt.scatter(amplitudes, v_ranges, 
                       label=stim_type,
                       color=self.stim_colors.get(stim_type, 'gray'),
                       alpha=0.6)
            
        plt.xlabel('Stimulus Amplitude (pA)')
        plt.ylabel('Voltage Range (mV)')
        plt.title('Stimulus Response Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
    def plot_response_heatmap(self, sweep_data: List[Dict]) -> None:
        """
        Create a heatmap of voltage responses across different sweeps.
        
        Args:
            sweep_data: List of dictionaries containing sweep information
        """
        # Prepare data for heatmap
        stim_types = sorted(set(s['stimulus_type'].decode() for s in sweep_data))
        data = []
        
        for stim_type in stim_types:
            sweeps = [s for s in sweep_data if s['stimulus_type'].decode() == stim_type]
            if sweeps:
                data.append({
                    'Stimulus Type': stim_type,
                    'Mean Response (mV)': np.mean([s['voltage_range'] for s in sweeps]),
                    'Count': len(sweeps),
                    'Mean Amplitude (pA)': np.mean([s['amplitude'] for s in sweeps])
                })
        
        # Create heatmap using seaborn
        plt.figure(figsize=(12, 8))
        data_array = np.array([[d['Mean Response (mV)'] for d in data]])
        sns.heatmap(data_array, 
                   xticklabels=[d['Stimulus Type'] for d in data],
                   yticklabels=['Mean Response'],
                   annot=True, fmt='.1f', cmap='viridis')
        plt.title('Mean Voltage Response by Stimulus Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
    def save_plot(self, filename: str) -> None:
        """
        Save the current plot to file.
        
        Args:
            filename: Name for the saved file
        """
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {filepath}")
        plt.close()