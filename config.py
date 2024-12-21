# neuron_analysis/config.py

class Config:
    SPIKE_THRESHOLD = -20  # mV
    WAVEFORM_WINDOW = 100  # points
    BASELINE_POINTS = 20
    MAX_SPIKES_PER_SWEEP = 50
    
    # Visualization parameters
    PLOT_STYLE = 'seaborn'
    RASTER_FIGSIZE = (10, 6)
    FEATURE_FIGSIZE = (12, 8)
    WAVEFORM_FIGSIZE = (10, 6)
    
    # Feature analysis
    BASELINE_POINTS = 20
    FEATURES_TO_ANALYZE = ['amplitude', 'width', 'ahp', 'max_dvdt']
    FEATURE_TITLES = {
        'amplitude': 'Amplitude (mV)',
        'width': 'Width (ms)',
        'ahp': 'AHP (mV)',
        'max_dvdt': 'Max dV/dt (mV/ms)'
    }
    
    # Cache settings
    CACHE_DIRECTORY = 'allen_cache'