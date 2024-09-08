import random
import shutil
import tkinter as tk
from json import load
from tkinter import ttk

import pandas as pd
from scipy.special import kl_div
from scipy.stats import pearsonr

from config import *
from scenarioCSS import *

ARCHIVE_PATH = "scenarios/css/mcarchive/"
ARCHIVE_DEFAULT_NAME = "MonteCarlo-DefaultRun"
ARCHIVE_NAME = "MonteCarlo-Run"
ARCHIVE_EXTENSION = ".json"

SENSORS_RANGE = range(1,9)

# Monte Carlo Simulation Class
class MonteCarloCSS:
    def __init__(self, params, sensors_val):
        """Constructor"""
        self.use_css_constellation = params["use_css_constellation"]
        self.use_eclipse = params["use_eclipse"]
        self.use_kelly = params["use_kelly"]
        self.number_of_cycles = params["number_of_cycles"]
        self.number_of_sensors = sensors_val
        self.is_tampered_fov = params["is_tampered_fov"]
        self.is_tampered_scale_factor = params["is_tampered_scale_factor"]

    def runDefault(self):
        """Run the default simulation"""
        run(
            use_css_constellation=self.use_css_constellation,
            use_eclipse=self.use_eclipse,
            use_kelly=self.use_kelly,
            number_of_cycles=self.number_of_cycles,
            number_of_sensors=self.number_of_sensors,
            is_archive=True,
            archive_name=ARCHIVE_PATH + ARCHIVE_DEFAULT_NAME + ARCHIVE_EXTENSION,
        )

    def getRandomParams(self):
        """Generate random parameters for the sensors"""
        random.seed()
        randomparams = []
        for i in range(max(3, self.number_of_sensors)):
            secured_default_index = min(i, len(DEFAULT_CSS_CONFIG["params"]) - 1)

            if self.is_tampered_fov:
                fov = random.uniform(
                    TAMPERED_RANGES[0]["range_min"]["fov"],
                    TAMPERED_RANGES[0]["range_max"]["fov"],
                )
            else:
                fov = DEFAULT_CSS_CONFIG["params"][secured_default_index]["fov"]


            r_B = DEFAULT_CSS_CONFIG["params"][secured_default_index]["r_B"]

            if self.is_tampered_scale_factor:
                scaleFactor = random.uniform(
                    TAMPERED_RANGES[0]["range_min"]["scaleFactor"],
                    TAMPERED_RANGES[0]["range_max"]["scaleFactor"],
                )
            else:
                scaleFactor = DEFAULT_CSS_CONFIG["params"][secured_default_index][
                    "scaleFactor"
                ]

            randomparams.append(
                {
                    "fov": fov,
                    "r_B": r_B,
                    "scaleFactor": scaleFactor,
                }
            )

        return randomparams

    def runMonteCarlo(self):
        """Run the Monte Carlo simulation"""
        for i in range(NUMBER_OF_RUNS):
            tampered_params = self.getRandomParams()
            run(
                use_css_constellation=self.use_css_constellation,
                use_eclipse=self.use_eclipse,
                use_kelly=self.use_kelly,
                number_of_cycles=self.number_of_cycles,
                number_of_sensors=self.number_of_sensors,
                sensor_params=tampered_params,
                is_archive=True,
                archive_name=ARCHIVE_PATH + ARCHIVE_NAME + str(i) + ARCHIVE_EXTENSION,
            )

    def getCalculations(self):
        """Get the Pearson MSE between the default run and each Monte Carlo runs"""
        # open the default run archive
        with open(ARCHIVE_PATH + ARCHIVE_DEFAULT_NAME + ARCHIVE_EXTENSION, "r") as f:
            default_data = load(f)

        default_y = np.array(default_data["y"])

        # open all the monte carlo run archives
        montecarlo_data = []
        for i in range(NUMBER_OF_RUNS):
            with open(
                ARCHIVE_PATH + ARCHIVE_NAME + str(i) + ARCHIVE_EXTENSION, "r"
            ) as f:
                montecarlo_data.append(load(f))

        if default_y.ndim > 1:
            default_y = default_y.flatten()


        # compare the default run with each monte carlo run
        mse_values = []
        for idx, data in enumerate(montecarlo_data):
            y = np.array(data["y"])

            if np.array(y).ndim > 1:
                y = y.flatten()

            # calculate the Mean Squared Error
            mse = np.mean((default_y - y) ** 2)
            mse_values.append(mse)

        return mse_values

    def plotResults(self, montecarlo_data, title="CSS Signals over Time"):
        """Plot the results of the default run and the Monte Carlo runs"""
        fig = go.Figure()

        data_x = np.array(montecarlo_data["x"])
        data_y = np.array(montecarlo_data["y"])

        for idx in range(self.number_of_sensors):
            fig.add_trace(
                go.Scatter(
                    x=data_x[idx],
                    y=data_y[idx],
                    mode="lines",
                    name=f"CSS_{idx}",
                )
            )

        fig.update_layout(
            title=f"{title}",
            xaxis_title="Time [sec]",
            yaxis_title="CSS Signals",
            legend_title="Sensors",
        )

        fig.show()

    def getKLDivergence(self):
        """Get the KL Divergence between the default run and each Monte Carlo runs"""
        # open the default run archive
        with open(ARCHIVE_PATH + ARCHIVE_DEFAULT_NAME + ARCHIVE_EXTENSION, "r") as f:
            default_data = load(f)

        default_y = np.array(default_data["data"][0]["y"])

        # open all the monte carlo run archives
        montecarlo_data = []
        for i in range(NUMBER_OF_RUNS):
            with open(
                ARCHIVE_PATH + ARCHIVE_NAME + str(i) + ARCHIVE_EXTENSION, "r"
            ) as f:
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
    """Display the results in a table"""
    root = tk.Tk()
    root.title(title)

    tree = ttk.Treeview(root)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)

    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill="both")

    root.mainloop()


if __name__ == "__main__":
    # Create the archive folder
    if not os.path.exists(ARCHIVE_PATH):
        os.makedirs(ARCHIVE_PATH)

    results = []

    for parameters in SIMULATIONS_PARAMETERS:
        for sensors_val in SENSORS_RANGE:
            monteCarlo = MonteCarloCSS(parameters, sensors_val)
            monteCarlo.runDefault()
            monteCarlo.runMonteCarlo()
            results.append(monteCarlo.getCalculations())

    data = {
        "use_css_constellation": [],
        "use_eclipse": [],
        "use_kelly": [],
        "number_of_cycles": [],
        "number_of_sensors": [],
        "is_tampered_fov": [],
        "is_tampered_scale_factor": [],
    }

    for params in SIMULATIONS_PARAMETERS:
        data["use_css_constellation"].append(params["use_css_constellation"])
        data["use_eclipse"].append(params["use_eclipse"])
        data["use_kelly"].append(params["use_kelly"])
        data["number_of_cycles"].append(params["number_of_cycles"])
        data["number_of_sensors"].append(SENSORS_RANGE)
        data["is_tampered_fov"].append(params["is_tampered_fov"])
        data["is_tampered_scale_factor"].append(params["is_tampered_scale_factor"])

    simulation_df = pd.DataFrame(data)

    min_mse_values = []
    max_mse_values = []
    avg_mse_values = []
    std_mse_values = []

    for idx, mse_values in enumerate(results):
        # get the min, max, average and standard deviation of the MSE factors for each simulation
        min_mse_values.append(min(mse_values))
        max_mse_values.append(max(mse_values))
        avg_mse_values.append(sum(mse_values) / len(mse_values))
        std_mse_values.append(pd.Series(mse_values).std())

    data = {
        "min_mse_values": min_mse_values,
        "max_mse_values": max_mse_values,
        "avg_mse_values": avg_mse_values,
        "std_mse_values": std_mse_values,
    }

    mse_df = pd.DataFrame(data)



    fig = go.Figure()

    results = np.array(results) 
    # Loop over simulations and add them to the plot
    for sim_idx in range(len(SIMULATIONS_PARAMETERS)):
        # Extract the rows for the current simulation
        sensors = len(SENSORS_RANGE)
        sim_mse = results[sim_idx * sensors:(sim_idx + 1) * (sensors), :]
        
        # Calculate mean and standard deviation for each sensor count across repetitions
        sim_mean = np.mean(sim_mse, axis=1)  # Mean MSE for current simulation
        sim_std = np.std(sim_mse, axis=1)    # Std deviation for current simulation
        
        # Add trace for the current simulation
        fig.add_trace(go.Scatter(
            x=list(SENSORS_RANGE), 
            y=sim_mean, 
            mode='lines+markers', 
            name=SIMULATIONS_PARAMETERS[sim_idx]["name"],
            error_y=dict(
                type='data', 
                array=sim_std,  # Standard deviation as error bars
                visible=True,
                thickness=1.5,
                width=3,
            ),
            # line=dict(
            #     color=f'rgba({50 * sim_idx}, {100 + 50 * sim_idx}, {150 + 50 * sim_idx}, 1)'
            # )  # Dynamic coloring for each simulation
        ))

    # Set plot title and labels
    fig.update_layout(
        title='MSE with Error Bars Across CSS Sensors',
        xaxis_title='Number of Sensors',
        yaxis_title='Mean Squared Error (%)',
        template="plotly_white",
    )

    # Save the plot as a PNG
    fig.write_image(f"mse_error_bars_multiple_simulations_plot_runs={NUMBER_OF_RUNS}.png")

    # displayBoard(simulation_df, "Simulations Parameters")
    displayBoard(mse_df, "MSE Factors")

    # Delete the archive folder
    shutil.rmtree(ARCHIVE_PATH)


