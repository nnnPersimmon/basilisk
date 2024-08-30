import random
from scenarioCSS import *
from config import *
from json import load
from scipy.stats import pearsonr
from scipy.special import kl_div
import pandas as pd
import tkinter as tk
from tkinter import ttk

ARCHIVE_PATH = "scenarios/css/mcarchive/"
ARCHIVE_DEFAULT_NAME = "MonteCarlo-DefaultRun"
ARCHIVE_NAME = "MonteCarlo-Run"
ARCHIVE_EXTENSION = ".json"

# Monte Carlo Simulation Class
class MonteCarloCSS:
 
    def __init__(self, params):
        """ Constructor """
        self.use_css_constellation = params["use_css_constellation"]
        self.use_platform = params["use_platform"]
        self.use_eclipse = params["use_eclipse"]
        self.use_kelly = params["use_kelly"]
        self.number_of_cycles = params["number_of_cycles"]
        self.number_of_sensors = params["number_of_sensors"]

    def runDefault(self):
        """ Run the default simulation """
        run(
            use_css_constellation=self.use_css_constellation,
            use_platform=self.use_platform,
            use_eclipse=self.use_eclipse,
            use_kelly=self.use_kelly,
            number_of_cycles=self.number_of_cycles,
            number_of_sensors=self.number_of_sensors,
            is_archive=True,
            archive_name=ARCHIVE_PATH + ARCHIVE_DEFAULT_NAME + ARCHIVE_EXTENSION,
        )

    def getRandomParams(self):
        """ Generate random parameters for the sensors """
        random.seed()
        randomparams = []
        for i in range(max(3, self.number_of_sensors)):
            fov = random.uniform(
                TAMPERED_RANGES[0]["range_min"]["fov"], 
                TAMPERED_RANGES[0]["range_max"]["fov"]
            )
            r_B = [
                random.uniform(
                    TAMPERED_RANGES[0]["range_min"]["r_B"][0], 
                    TAMPERED_RANGES[0]["range_max"]["r_B"][0]
                ),
                random.uniform(
                    TAMPERED_RANGES[0]["range_min"]["r_B"][1], 
                    TAMPERED_RANGES[0]["range_max"]["r_B"][1]
                ),
                random.uniform(
                    TAMPERED_RANGES[0]["range_min"]["r_B"][2], 
                    TAMPERED_RANGES[0]["range_max"]["r_B"][2]
                ),
            ]
            scaleFactor = random.uniform(
                TAMPERED_RANGES[0]["range_min"]["scaleFactor"], 
                TAMPERED_RANGES[0]["range_max"]["scaleFactor"]
            )
            
            randomparams.append({"fov": fov, "r_B": r_B, "scaleFactor": scaleFactor})
    
        return randomparams

    def runMonteCarlo(self):
        """ Run the Monte Carlo simulation """
        for i in range(NUMBER_OF_RUNS):
            tamptered_params = self.getRandomParams()
            run(
                use_css_constellation=self.use_css_constellation,
                use_platform=self.use_platform,
                use_eclipse=self.use_eclipse,
                use_kelly=self.use_kelly,
                number_of_cycles=self.number_of_cycles,
                number_of_sensors=self.number_of_sensors,
                sensor_params=tamptered_params,
                is_archive=True,
                archive_name=ARCHIVE_PATH + ARCHIVE_NAME + str(i) + ARCHIVE_EXTENSION,
            )

    def getPearsonCorrelation(self):
        """ Get the Pearson correlation between the default run and each Monte Carlo runs """
        # open the default run archive
        with open(ARCHIVE_PATH + ARCHIVE_DEFAULT_NAME + ARCHIVE_EXTENSION, 'r') as f:
            default_data = load(f)
        
        default_y = np.array(default_data["data"][0]["y"])
        

        # open all the monte carlo run archives
        montecarlo_data = []
        for i in range(NUMBER_OF_RUNS):
            with open(ARCHIVE_PATH + ARCHIVE_NAME + str(i) + ARCHIVE_EXTENSION, 'r') as f:
                montecarlo_data.append(load(f))
        
        # compare the default run with each monte carlo run
        correlations = []
        for data in montecarlo_data:
            y = np.array(data["data"][0]["y"])

            if default_y.ndim > 1:
                default_y = default_y.flatten()
            if np.array(y).ndim > 1:
                y = y.flatten()

            # calculate the Pearson correlation
            corr, _ = pearsonr(default_y, y)
            correlations.append(corr.item())
        
        return correlations
    
    def getKLDivergence(self):
        """ Get the KL Divergence between the default run and each Monte Carlo runs """
        # open the default run archive
        with open(ARCHIVE_PATH + ARCHIVE_DEFAULT_NAME + ARCHIVE_EXTENSION, 'r') as f:
            default_data = load(f)
        
        default_y = np.array(default_data["data"][0]["y"])
        
        # open all the monte carlo run archives
        montecarlo_data = []
        for i in range(NUMBER_OF_RUNS):
            with open(ARCHIVE_PATH + ARCHIVE_NAME + str(i) + ARCHIVE_EXTENSION, 'r') as f:
                montecarlo_data.append(load(f))
        
        # compare the default run with each monte carlo run
        divergences = []
        for data in montecarlo_data:
            y = np.array(data["data"][0]["y"])

            if default_y.ndim > 1:
                default_y = default_y.flatten()
            if np.array(y).ndim > 1:
                y = y.flatten()

            # calculate the KL divergence
            divergence = kl_div(default_y, y)
            divergences.append(divergence)
        
        return divergences


def displayBoard(df, title="Results"):
    """ Display the results in a table"""
    root = tk.Tk()
    root.title(title)

    tree = ttk.Treeview(root)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)

    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill='both')

    root.mainloop()

if __name__ == "__main__":

    correlations = []
    
    for parameters in SIMULATIONS_PARAMETERS:
        monteCarlo = MonteCarloCSS(parameters)
        monteCarlo.runDefault()
        monteCarlo.runMonteCarlo()
        correlations.append(monteCarlo.getPearsonCorrelation())
    
    data = {
        'use_css_constellation': [],
        'use_platform': [],
        'use_eclipse': [],
        'use_kelly': [],
        'number_of_cycles': [],
        'number_of_sensors': [],
    }

    for params in SIMULATIONS_PARAMETERS:
        data['use_css_constellation'].append(params["use_css_constellation"])
        data['use_platform'].append(params["use_platform"])
        data['use_eclipse'].append(params["use_eclipse"])
        data['use_kelly'].append(params["use_kelly"])
        data['number_of_cycles'].append(params["number_of_cycles"])
        data['number_of_sensors'].append(params["number_of_sensors"])

    
    simulation_df = pd.DataFrame(data)

    min_corr = []
    max_corr = []
    avg_corr = []
    std_corr = []
    
    for corr in correlations:
        # get the min, max, average and standard deviation of the correlation factors for each simulation
        min_corr.append(min(corr))
        max_corr.append(max(corr))
        avg_corr.append(sum(corr) / len(corr))
        std_corr.append(pd.Series(corr).std())

    data = {
        'min_corr': min_corr,
        'max_corr': max_corr,
        'avg_corr': avg_corr,
        'std_corr': std_corr,
    }

    corr_df = pd.DataFrame(data)


    #displayBoard(simulation_df, "Simulations Parameters")
    displayBoard(corr_df, "Correlation Factors")
