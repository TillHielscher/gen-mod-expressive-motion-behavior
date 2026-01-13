import numpy as np
from scipy.interpolate import make_interp_spline


class Phase:
    def __init__(self) -> None:        
        self.spline_linear = self.calculate_spline(["moderate", "moderate", "moderate"])
        
        self.spline = None
        self.progression_rates = ["moderate", "moderate", "moderate"]        
        self.set_phase()
        self.phase = self.phase_spline

    def set_phase(self, progression_rates=None, p_slow=False):
        if progression_rates is not None:
            self.progression_rates=progression_rates
        
        if p_slow:                
            progression_rates = ["slow"] + self.progression_rates + ["slow"]            
        else:
            progression_rates = self.progression_rates
        self.spline = self.calculate_spline(progression_rates) 


    def phase_spline(self, t, tau):
        # Calculate time progress
        progress = t/tau

        # Get the phase variable of the DMP based on the spline that was computed previously
        x = self.spline(progress)

        # Return the phase variable
        return x
    
    def phase_linear(self, t, tau):
        # Calculate time progress
        progress = t/tau

        # Get the phase variable of the DMP based on the spline that was computed previously
        x = self.spline_linear(progress)

        # Return the phase variable
        return x

    def calculate_spline(self, progression_rates):

        # Define the relative weights for "slow", "moderate", and "fast"
        rate_weights = {
            "slow": 1.0,
            "moderate": 2.0,
            "fast": 3.0
        }

        # Calculate the total weight
        total_weight = sum(rate_weights[rate] for rate in progression_rates)

        # Calculate the step sizes for each rate so that the total steps sum to 1.0
        step_values = {
            rate: (rate_weights[rate] / total_weight) for rate in rate_weights}

        # Initialize the list to store progress values at interval borders
        ordinates = [1.0]  # Start with a progress of 1.0 (fully completed)

        # Calculate the progress for each interval
        for rate in progression_rates:
            step = step_values[rate]
            last_value = ordinates[-1]
            new_value = max(0, last_value - step)
            ordinates.append(new_value)

        # Adjust the last value to be exactly 0.0 if it is not due to floating point precision
        ordinates[-1] = 0.0

        # Calculate corresponding times between 0 and 1
        num_intervals = len(progression_rates)
        abscissas = [i / num_intervals for i in range(num_intervals + 1)]

        # Create the spline
        spline = make_interp_spline(abscissas, ordinates, k=2)

        # Set the class variable such that the spline is accessible
        return spline