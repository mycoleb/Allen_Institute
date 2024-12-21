"""
Main script for analyzing neuronal action potentials.
This script loads neuronal recordings from the Allen Cell Types Database 
and analyzes their electrical activity patterns.

The analysis pipeline:
1. Loads a recording from a real neuron
2. Detects action potentials (spikes) in the voltage trace
3. Analyzes the shape and features of each spike
4. Creates visualizations to help understand the neuron's behavior
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt  # Added this import that was missing before

# Allen SDK imports for accessing neuronal data
from allensdk.core.cell_types_cache import CellTypesCache

# Import our analysis and visualization tools
from analyzers.action_potential import ActionPotentialAnalyzer
from visualizers.visualizer import APVisualizer
from config import Config

def setup_logging() -> None:
    """
    Sets up logging to track the analysis process.
    Creates both a file log and console output for better monitoring.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Set up file logging
    logging.basicConfig(
        filename=log_dir / 'neuron_analysis.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console output for immediate feedback
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info("Starting neuronal analysis")

def load_example_data() -> Dict[str, Any]:
    """
    Loads a sample neuron recording from the Allen Cell Types Database.
    Converts units to standard neurophysiology conventions:
    - Voltage: V -> mV
    - Current: A -> pA
    """
    try:
        # Initialize connection to Allen Cell Types Database
        manifest_path = Path('manifest.json')
        ctc = CellTypesCache(manifest_file=str(manifest_path))
        
        # Get cells
        cells = ctc.get_cells()
        if not cells:
            raise ValueError("No cells found in database")
        
        cell_id = cells[0]['id']
        dataset = ctc.get_ephys_data(cell_id)
        sweep_numbers = dataset.get_sweep_numbers()
        
        # Find a sweep with action potentials
        # Look for short square pulses which typically evoke spikes
        for sweep_number in sweep_numbers:
            sweep_data = dataset.get_sweep(sweep_number)
            metadata = dataset.get_sweep_metadata(sweep_number)
            
            # Convert units
            v = sweep_data['response'] * 1000.0  # Convert to mV
            i = sweep_data['stimulus'] * 1e12    # Convert to pA
            
            # Get the stimulus period
            stim_start_idx, stim_end_idx = sweep_data['index_range']
            
            # Check if this sweep likely contains spikes
            v_during_stim = v[stim_start_idx:stim_end_idx]
            v_range = np.ptp(v_during_stim)
            
            if v_range > 20:  # Looking for >20mV deflections (typical for spikes)
                logging.info(f"\nFound promising sweep {sweep_number}:")
                logging.info(f"Stimulus type: {metadata['aibs_stimulus_name']}")
                logging.info(f"Stimulus amplitude: {metadata['aibs_stimulus_amplitude_pa']:.1f} pA")
                logging.info(f"Voltage range during stimulus: {v_range:.1f} mV")
                
                return {
                    'response': v,                    # mV
                    'stimulus': i,                    # pA
                    'sampling_rate': sweep_data['sampling_rate'],
                    'sweep_number': sweep_number,
                    'cell_id': cell_id,
                    'stim_start': stim_start_idx,
                    'stim_end': stim_end_idx,
                    'metadata': metadata
                }
        
        raise ValueError("No sweeps with significant voltage deflections found")
        
    except Exception as e:
        logging.error(f"Failed to load example data: {str(e)}")
        raise
def explore_dataset() -> None:
    """
    Explores the Allen Cell Types Database dataset to identify sweeps with neural activity.
    Prints summary statistics for all available sweeps.
    """
    try:
        # Initialize connection to Allen Cell Types Database
        manifest_path = Path('manifest.json')
        ctc = CellTypesCache(manifest_file=str(manifest_path))
        
        # Get cells and basic info
        cells = ctc.get_cells()
        logging.info(f"\nFound {len(cells)} total cells in database")
        
        # Get data for our cell
        cell_id = cells[0]['id']
        dataset = ctc.get_ephys_data(cell_id)
        sweep_numbers = dataset.get_sweep_numbers()
        logging.info(f"\nExamining cell {cell_id}")
        logging.info(f"Found {len(sweep_numbers)} total sweeps")
        
        # Create a summary of all sweeps
        logging.info("\nAnalyzing all sweeps:")
        logging.info("Sweep\tStimulus Type\tAmplitude(pA)\tVoltage Range(mV)")
        logging.info("-" * 60)
        
        active_sweeps = []  # Keep track of sweeps with significant activity
        
        for sweep_number in sweep_numbers:
            sweep_data = dataset.get_sweep(sweep_number)
            metadata = dataset.get_sweep_metadata(sweep_number)
            
            # Convert units
            v = sweep_data['response'] * 1000.0  # Convert to mV
            i = sweep_data['stimulus'] * 1e12    # Convert to pA
            
            # Get the stimulus period
            stim_start_idx, stim_end_idx = sweep_data['index_range']
            v_during_stim = v[stim_start_idx:stim_end_idx]
            v_range = np.ptp(v_during_stim)
            
            # Log sweep info
            stim_type = metadata.get('aibs_stimulus_name', 'Unknown')
            stim_amp = metadata.get('aibs_stimulus_amplitude_pa', 0)
            
            logging.info(f"{sweep_number}\t{stim_type}\t{stim_amp:.1f}\t{v_range:.1f}")
            
            # Keep track of sweeps with significant voltage changes
            if v_range > 20:  # More than 20mV change suggests action potentials
                active_sweeps.append({
                    'sweep_number': sweep_number,
                    'stimulus_type': stim_type,
                    'amplitude': stim_amp,
                    'voltage_range': v_range
                })
        
        logging.info(f"\nFound {len(active_sweeps)} sweeps with significant activity")
        
        # Print summary of active sweeps
        if active_sweeps:
            logging.info("\nSweeps with neural activity:")
            logging.info("Sweep\tStimulus Type\tAmplitude(pA)\tVoltage Range(mV)")
            logging.info("-" * 60)
            for sweep in active_sweeps:
                logging.info(f"{sweep['sweep_number']}\t{sweep['stimulus_type']}\t"
                           f"{sweep['amplitude']:.1f}\t{sweep['voltage_range']:.1f}")
                
        return active_sweeps
        
    except Exception as e:
        logging.error(f"Failed to explore dataset: {str(e)}")
        raise

def main() -> None:
    """
    Main execution function that explores the dataset and creates visualizations.
    """
    try:
        # Set up logging
        setup_logging()
        
        # Load and analyze data
        logging.info("Starting exploration of neural recording data")
        
        # Initialize connection to Allen Cell Types Database
        manifest_path = Path('manifest.json')
        ctc = CellTypesCache(manifest_file=str(manifest_path))
        
        # Get cells and basic info
        cells = ctc.get_cells()
        logging.info(f"\nFound {len(cells)} total cells in database")
        
        # Get data for our cell
        cell_id = cells[0]['id']
        dataset = ctc.get_ephys_data(cell_id)
        sweep_numbers = dataset.get_sweep_numbers()
        logging.info(f"\nExamining cell {cell_id}")
        logging.info(f"Found {len(sweep_numbers)} total sweeps")
        
        # Create visualizer
        visualizer = APVisualizer(save_dir='neuron_plots')
        
        # Analyze sweeps and collect data
        active_sweeps = []
        
        logging.info("\nAnalyzing all sweeps:")
        logging.info("Sweep\tStimulus Type\tAmplitude(pA)\tVoltage Range(mV)")
        logging.info("-" * 60)
        
        for sweep_number in sweep_numbers:
            sweep_data = dataset.get_sweep(sweep_number)
            metadata = dataset.get_sweep_metadata(sweep_number)
            
            # Convert units
            v = sweep_data['response'] * 1000.0  # Convert to mV
            i = sweep_data['stimulus'] * 1e12    # Convert to pA
            
            # Get the stimulus period
            stim_start_idx, stim_end_idx = sweep_data['index_range']
            v_during_stim = v[stim_start_idx:stim_end_idx]
            v_range = np.ptp(v_during_stim)
            
            # Log sweep info
            stim_type = metadata.get('aibs_stimulus_name', 'Unknown')
            stim_amp = metadata.get('aibs_stimulus_amplitude_pa', 0)
            
            logging.info(f"{sweep_number}\t{stim_type}\t{stim_amp:.1f}\t{v_range:.1f}")
            
            # Track sweeps with significant activity
            if v_range > 20:  # More than 20mV change suggests action potentials
                active_sweeps.append({
                    'sweep_number': sweep_number,
                    'stimulus_type': stim_type,
                    'amplitude': stim_amp,
                    'voltage_range': v_range
                })
                
                # Create individual sweep plots for active sweeps
                t = np.arange(len(v)) / sweep_data['sampling_rate']
                visualizer.plot_sweep(t, v, i, 
                                    {'sweep_number': sweep_number, 
                                     'stimulus_type': stim_type})
                visualizer.save_plot(f'sweep_{sweep_number}.png')
        
        # Create summary visualizations
        logging.info(f"\nCreating summary plots for {len(active_sweeps)} active sweeps")
        
        # Plot stimulus comparison
        visualizer.plot_stimulus_comparison(active_sweeps)
        visualizer.save_plot('stimulus_comparison.png')
        
        # Plot response heatmap
        visualizer.plot_response_heatmap(active_sweeps)
        visualizer.save_plot('response_heatmap.png')
        
        logging.info("\nAll visualizations have been saved to the 'neuron_plots' directory")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()