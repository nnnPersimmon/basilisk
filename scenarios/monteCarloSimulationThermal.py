import inspect
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

filename = inspect.getframeinfo(inspect.currentframe()).filename
fileNameString = os.path.basename(os.path.splitext(__file__)[0])
path = os.path.dirname(os.path.abspath(filename))


from Basilisk import __path__

bskPath = __path__[0]


from Basilisk.architecture import messaging
from Basilisk.simulation import (sensorThermal)
from Basilisk.utilities import (macros)
from Basilisk.utilities.MonteCarlo.Controller import (Controller,
                                                      RetentionPolicy)
from Basilisk.utilities.MonteCarlo.Dispersions import (
    NormalVectorCartDispersion, UniformDispersion)

from scenarioSensorThermal import *

NUMBER_OF_RUNS = 10
VERBOSE = True

# Global lists to store data across runs
all_temp_data = []
tvec_global = None

fig = go.Figure()

# Here are the name of some messages that we want to retain or otherwise use
temperatureOutMsgName = "temperatureOutMsg"

# We also will need the simulationTime and samplingTimes
numDataPoints = 500
simulationTime = macros.min2nano(200)
samplingTime = simulationTime // (numDataPoints - 1)


def runMonteCarlo():
    mcController = Controller()

    mcController.setSimulationFunction(createScenarioSensorThermal)
    mcController.setExecutionFunction(executeScenario)
    mcController.setExecutionCount(NUMBER_OF_RUNS)
    mcController.setShouldDisperseSeeds(True)
    mcController.setShowProgressBar(True)
    # Optionally set the number of cores to use
    # mcController.setThreadCount(PROCESSES)
    mcController.setVerbose(VERBOSE)
    dirName = "montecarlo_test" + str(os.getpid())
    mcController.setArchiveDir(dirName)

    dispInitTemp = "thermalSensor.T_0"
    dispOrientation = "thermalSensor.nHat_B"
    dispPowerDraw = "thermalSensor.sensorPowerDraw"
    dispList = [dispInitTemp, dispOrientation, dispPowerDraw]

    # Add dispersions with their dispersion type
    mcController.addDispersion(
        UniformDispersion(dispInitTemp, ([0.0 - 1.0 * 100.0, 0.0 + 1.0 * 100.0]))
    )
    mcController.addDispersion(
        NormalVectorCartDispersion(dispOrientation, [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    )
    mcController.addDispersion(
        UniformDispersion(dispPowerDraw, ([30.0 - 0.5 * 30.0, 30.0 + 1.0 * 100.0]))
    )

    # Add retention policy
    retentionPolicy = RetentionPolicy()
    retentionPolicy.addMessageLog(temperatureOutMsgName, ["temperature"])
    retentionPolicy.setDataCallback(plotSim)
    mcController.addRetentionPolicy(retentionPolicy)

    NormalMCRun(mcController, dirName, dispList)

    # Now we clean up data from this test
    shutil.rmtree(dirName)
    assert not os.path.exists(dirName), "No leftover data should exist after the test"

    plotConfidenceIntervals()


def createScenarioSensorThermal():

    scSim = setup_simulation()

    scObject = setup_spacecraft(scSim)
    scSim.hubref = scObject.hub

    gravFactory, mu = setup_grav_body(scObject)

    spiceObject = setup_spice_interface(gravFactory, scSim)

    rN, vN, n = setup_orbit(mu)

    mrpControl, locPoint, P = setup_modules(scSim, scObject, spiceObject, n, rN, vN)

    params = {
        "T_0": 0,  # Celsius
        "nHat_B": [0, 0, 1],
        "sensorArea": 1.0,  # m^2
        "sensorAbsorptivity": 0.25,
        "sensorEmissivity": 0.34,
        "sensorMass": 2.0,  # kg
        "sensorSpecificHeat": 890,
        "sensorPowerDraw": 30.0,  # W
    }

    # temperatureOutMsg, sensorThermal = setup_thermal_sensor(spiceObject, scObject, scSim, params)

    # Now add the thermal sensor module
    thermalSensor = sensorThermal.SensorThermal()
    # Apply parameters from the configuration
    thermalSensor.T_0 = params.get(
        "T_0", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["T_0"]
    )
    thermalSensor.nHat_B = params.get(
        "nHat_B", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["nHat_B"]
    )
    thermalSensor.sensorArea = params.get(
        "sensorArea", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorArea"]
    )
    thermalSensor.sensorAbsorptivity = params.get(
        "sensorAbsorptivity",
        DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorAbsorptivity"],
    )
    thermalSensor.sensorEmissivity = params.get(
        "sensorEmissivity", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorEmissivity"]
    )
    thermalSensor.sensorMass = params.get(
        "sensorMass", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorMass"]
    )
    thermalSensor.sensorSpecificHeat = params.get(
        "sensorSpecificHeat",
        DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorSpecificHeat"],
    )
    thermalSensor.sensorPowerDraw = params.get(
        "sensorPowerDraw", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorPowerDraw"]
    )

    thermalSensor.sunInMsg.subscribeTo(spiceObject.planetStateOutMsgs[0])
    thermalSensor.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.thermalSensor = thermalSensor
    scSim.msgRecList = {}
    scSim.AddModelToTask(simTaskName, thermalSensor)
    scSim.msgRecList[temperatureOutMsgName] = thermalSensor.temperatureOutMsg.recorder(
        samplingTime
    )
    scSim.AddModelToTask(simTaskName, scSim.msgRecList[temperatureOutMsgName])

    # Create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = intertia
    configDataMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
    mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)

    scSim.additionalReferences = [mrpControl, locPoint, P, thermalSensor]

    return scSim


def executeScenario(sim):

    #   retrieve the module references
    locPoint = sim.additionalReferences[1]
    P = sim.additionalReferences[2]

    #   initialize Simulation
    sim.InitializeSimulation()

    #   configure a simulation stop time and execute the simulation run
    sim.ConfigureStopTime(macros.sec2nano(int(P)))
    # Begin the simulation time run set above
    sim.ExecuteSimulation()

    # Change the location pointing vector and run the sim for another period
    locPoint.pHat_B = [0, 0, -1]
    sim.ConfigureStopTime(macros.sec2nano(int(2 * P)))  # seconds to stop simulation
    sim.ExecuteSimulation()


# This method is used to plot the retained data of a simulation.
# It is called once for each run of the simulation, overlapping the plots


def plotSim(data, retentionPolicy):
    global all_temp_data, tvec_global
    # Retrieve the logged data
    tempData = data["messages"][temperatureOutMsgName + ".temperature"][:, 1:]
    tvec = (
        data["messages"][temperatureOutMsgName + ".temperature"][:, 0]
        * macros.NANO2HOUR
    )
    np.set_printoptions(precision=16)

    # Store the data for confidence interval calculations
    all_temp_data.append(tempData.flatten())
    if tvec_global is None:
        tvec_global = tvec * 60.0  # Convert to minutes and store globally

    # Add a new trace to the global figure for each run
    fig.add_trace(
        go.Scatter(
            x=tvec * 60.0,
            y=tempData.flatten(),  # Flatten in case tempData is a 2D array with a single column
            mode="lines",
            name=f'Run {str(data["index"])}',  # Label the trace with the run index
        )
    )

    # Customize layout (only once, so it's efficient)
    fig.update_layout(
        title="Temperature Over Time",
        xaxis_title="Time (min)",
        yaxis_title="Temperature (deg C)",
        legend_title="Run Index",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    # Optionally save the figure as an image file (e.g., PNG)
    fig.write_image(
        "temperature_plot.png"
    )  # Save the combined plot after all runs are added


def plotConfidenceIntervals():
    global all_temp_data, tvec_global

    # Convert the list of arrays into a 2D numpy array for easier statistics
    all_temp_data = np.array(all_temp_data)

    # Calculate mean and standard deviation across all runs
    mean_temp = np.mean(all_temp_data, axis=0)
    std_temp = np.std(all_temp_data, axis=0)
    # Define confidence intervals (e.g., 95%)
    lower_bound = mean_temp - 1.96 * std_temp
    upper_bound = mean_temp + 1.96 * std_temp

    # Plot mean temperature with confidence intervals
    fig_conf = go.Figure()

    # Plot the mean temperature line
    fig_conf.add_trace(
        go.Scatter(
            x=tvec_global,
            y=mean_temp,
            mode="lines",
            name="Mean Temperature",
            line=dict(color="blue"),
        )
    )

    # Plot the confidence interval as a shaded area
    fig_conf.add_trace(
        go.Scatter(
            x=np.concatenate(
                [tvec_global, tvec_global[::-1]]
            ),  # x values (forward and backward)
            y=np.concatenate(
                [upper_bound, lower_bound[::-1]]
            ),  # upper bound first, then lower bound
            fill="toself",
            fillcolor="rgba(0, 100, 80, 0.2)",  # semi-transparent fill color
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval",
        )
    )

    # Customize layout
    fig_conf.update_layout(
        title=f"Monte Carlo Simulations (N={len(all_temp_data)})",
        xaxis_title="Time (min)",
        yaxis_title="Temperature (deg C)",
        legend_title="Legend",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    # Save or show the figure
    fig_conf.write_image(
        f"temperature_confidence_intervals_simulations={len(all_temp_data)}.png"
    )
    # pio.show(fig_conf)

    return fig_conf


def InitialConditionRun(mcController):
    # Now run initial conditions
    icName = path + "/Support/run_MC_IC"
    mcController.setICDir(icName)
    mcController.setICRunFlag(True)
    numberICs = 3
    mcController.setExecutionCount(numberICs)

    # Rerunning the case shouldn't fail
    runsList = list(range(numberICs))
    failed = mcController.runInitialConditions(runsList)
    assert len(failed) == 0, "Should run ICs successfully"

    # monteCarlo.executeCallbacks([4,6,7])
    runsList = list(range(numberICs))
    mcController.executeCallbacks(runsList)

    # And show the plots
    plt.show()
    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    # Now we clean up data from this test
    os.remove(icName + "/" + "MonteCarlo.data")
    for i in range(numberICs):
        os.remove(icName + "/" + "run" + str(i) + ".data")
    assert not os.path.exists(
        icName + "/" + "MonteCarlo.data"
    ), "No leftover data should exist after the test"


def NormalMCRun(mcController, dirName, dispList):
    # After the monteCarlo run is configured, it is executed.
    # This method returns the list of jobs that failed.
    failures = mcController.executeSimulations()

    assert len(failures) == 0, "No runs should fail"

    # Now in another script (or the current one), the data from this simulation can be easily loaded.
    # This demonstrates loading it from disk
    monteCarloLoaded = Controller.load(dirName)

    # Then retained data from any run can then be accessed in the form of a dictionary
    # with two sub-dictionaries for messages and variables:
    retainedData = monteCarloLoaded.getRetainedData(NUMBER_OF_RUNS - 1)
    assert retainedData is not None, "Retained data should be available after execution"
    assert "messages" in retainedData, "Retained data should retain messages"

    # We also can rerun a case using the same parameters and random seeds
    # If we rerun a properly set-up run, it should output the same data.
    # Here we test that if we rerun the case the data doesn't change
    oldOutput = retainedData["messages"][temperatureOutMsgName + ".temperature"]

    # Rerunning the case shouldn't fail
    failed = monteCarloLoaded.reRunCases([NUMBER_OF_RUNS - 1])
    assert len(failed) == 0, "Should rerun case successfully"

    # Now access the newly retained data to see if it changed
    retainedData = monteCarloLoaded.getRetainedData(NUMBER_OF_RUNS - 1)
    newOutput = retainedData["messages"][temperatureOutMsgName + ".temperature"]
    for k1, v1 in enumerate(oldOutput):
        for k2, v2 in enumerate(v1):
            assert (
                math.fabs(oldOutput[k1][k2] - newOutput[k1][k2]) < 0.001
            ), "Outputs shouldn't change on runs if random seeds are same"

    # We can also access the initial parameters
    # The random seeds should differ between runs, so we will test that
    params1 = monteCarloLoaded.getParameters(NUMBER_OF_RUNS - 1)
    params2 = monteCarloLoaded.getParameters(NUMBER_OF_RUNS - 2)
    assert (
        "TaskList[0].TaskModels[0].RNGSeed" in params1
    ), "random number seed should be applied"
    for dispName in dispList:
        assert dispName in params1, "dispersion should be applied"
        # assert two different runs had different parameters.
        assert (
            params1[dispName] != params2[dispName]
        ), "dispersion should be different in each run"

    # Now we execute our callback for the retained data.
    # For this run, that means executing the plot.
    # We can plot only runs 4,6,7 overlapped
    # monteCarloLoaded.executeCallbacks([4,6,7])
    # or execute the plot on all runs
    monteCarloLoaded.executeCallbacks()


if __name__ == "__main__":
    runMonteCarlo()

# TODO: create simulation that links the thermal sensor to other subsystems so we can quantify activity time of sat
# - find context of thermal sensor and subsytems
# - maybe test with another type of sensors (startracker)

# TODO: create simulations of define attacks scenarios;
# - abrupt attack
# - gradual increase of intensity (it incorportates time)

# TODO: Monte Carlo analysis: change figures layout; conf intervals and max/min dev

# TODO: define range of parameters that should be changed
# - autonomous decision on the sat
# - human in the loop to check to gnd station. see if can get some inferences
# - attack vs dysfunction

# TODO: define countermeasures ; diversity, reliability and robustness
# - detection : depends on sampling time and attack threshold taht could be tuned
# - maybe some assumptions; prior information on % of attack (const) and/or number of sensors
# - showcase benefit of installing the system in respect to activity time of sat
