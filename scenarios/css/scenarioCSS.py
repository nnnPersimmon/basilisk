import os
import click
from json import dumps
import numpy as np
import plotly.graph_objects as go
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.simulation import coarseSunSensor, spacecraft
from Basilisk.utilities import (  # general support file with common unit test functions
    SimulationBaseClass,
    macros,
)
from Basilisk.utilities import orbitalMotion as om
from Basilisk.utilities import (  # general support file with common unit test functions
    unitTestSupport,
    vizSupport,
)
from config import DEFAULT_CSS_CONFIG

TASK_NAME = "css_simulation"

# TODO: add use Kelly
# TODO: check feasibility of theta? for many sensors


@click.command()
@click.option(
    "--use-css-constellation", is_flag=True, default=False, help="Use CSS Constellation"
)
@click.option("--use-platform", is_flag=True, default=False, help="Use Platform")
@click.option("--use-eclipse", is_flag=True, default=False, help="Use Eclipse")
@click.option("--use-kelly", is_flag=True, default=False, help="Use Kelly")
@click.option(
    "--number-of-cycles",
    type=int,
    default=5,
    help="Number of cycles (must be 1 or more)",
)
@click.option(
    "--number-of-sensors",
    type=int,
    default=3,
    help="Number of CSS sensors (must be 1 or more)",
)
def run_click(
    use_css_constellation,
    use_platform,
    use_eclipse,
    use_kelly,
    number_of_cycles,
    number_of_sensors,
):
    run(
        use_css_constellation,
        use_platform,
        use_eclipse,
        use_kelly,
        number_of_cycles,
        number_of_sensors,
    )

def run(
    use_css_constellation,
    use_platform,
    use_eclipse,
    use_kelly,
    number_of_cycles,
    number_of_sensors,
    sensor_params=DEFAULT_CSS_CONFIG["params"],
    is_archive=False,
    archive_name="defaultName",
):
    """
    Run the simulation with the specified parameters.

    Args:
        use_css_constellation (bool): Flag indicating if the CSS Constellation should be used.
        use_platform (bool): Flag specifying if the Platform should be used.
        use_eclipse (bool): Flag indicating if the Eclipse input message should be used.
        use_kelly (bool): Flag specifying if the Kelly factor should be used.
        number_of_cycles (int): Number of cycles to run the simulation (must be 1 or more).
        number_of_sensors (int): Number of CSS sensors to use in the simulation (must be 1 or more).
        sensor_params (list): List of dictionaries containing the parameters for the CSS sensors.
        is_archive (bool): Flag indicating if the results should be archived.
        archive_name (str): Name of the archive file to store the results.
    """

    scSim = create_simulation()

    simulationTime = set_simulation_time(300 * number_of_cycles)
    create_simulation_process(scSim, TASK_NAME, 1.0)

    scObject = setup_spacecraft()
    add_spacecraft_to_simulation(scSim, TASK_NAME, scObject)

    sunPositionMsg = create_sun_position_message()

    if use_eclipse:
        eclipseMsg = create_eclipse_message()

    def setup_css(CSS):
        CSS.maxOutput = 2.0
        CSS.minOutput = 0.5
        CSS.sunInMsg.subscribeTo(sunPositionMsg)
        CSS.stateInMsg.subscribeTo(scObject.scStateOutMsg)

        if use_eclipse:
            CSS.sunEclipseInMsg.subscribeTo(eclipseMsg)
        if use_platform:
            CSS.setBodyToPlatformDCM(90.0 * macros.D2R, 0.0, 0.0)
            CSS.theta = -90.0 * macros.D2R
            CSS.phi = 0 * macros.D2R
            CSS.setUnitDirectionVectorWithPerturbation(0.0, 0.0)
        else:
            CSS.nHat_B = np.array([1.0, 0.0, 0.0])

    css_sensors = []
    for i in range(number_of_sensors):
        CSS = coarseSunSensor.CoarseSunSensor()
        CSS.ModelTag = f"CSS{i+1}_sensor"
        setup_css(CSS)
        CSS.fov = sensor_params[i]["fov"] * macros.D2R
        CSS.r_B = sensor_params[i]["r_B"]
        CSS.scaleFactor = sensor_params[i]["scaleFactor"]
        if i >= 1:
            CSS.CSSGroupID = i - 1
        # Configure specific attributes for each sensor if needed
        if i == 1:
            CSS.CSSGroupID = 0
            if use_platform:
                CSS.theta = 0.0 * macros.D2R
                CSS.setUnitDirectionVectorWithPerturbation(0.0, 0.0)
            else:
                CSS.nHat_B = np.array([0.0, 1.0, 0.0])
        elif i == 2:
            CSS.CSSGroupID = 1
            if use_platform:
                CSS.theta = 90.0 * macros.D2R
                CSS.setUnitDirectionVectorWithPerturbation(0.0, 0.0)
            else:
                CSS.nHat_B = np.array([-1.0, 0.0, 0.0])

        css_sensors.append(CSS)

    data_arrays = []
    cssLogs = []
    if use_css_constellation:
        # If instead of individual CSS a cluster of CSS units is to be evaluated as one,
        # then they can be grouped into a list, and added to the Basilisk execution
        # stack as a single entity.  This is done with
        cssArray = coarseSunSensor.CSSConstellation()
        cssArray.ModelTag = "css_array"
        cssArray.sensorList = coarseSunSensor.CSSVector(css_sensors)
        scSim.AddModelToTask(TASK_NAME, cssArray)
        # Here the CSSConstellation() module will call the individual CSS
        # update functions, collect all the sensor
        # signals, and store the output in a single output message
        # containing an array of CSS sensor signals.

        #
        #   Setup data logging before the simulation is initialized
        #
        cssConstLog = cssArray.constellationOutMsg.recorder()
        scSim.AddModelToTask(TASK_NAME, cssConstLog)
    else:
        # In this scenario (A) setup the CSS unit are each evaluated separately through
        # This means that each CSS unit creates a individual output messages.
        for css in css_sensors:
            scSim.AddModelToTask(TASK_NAME, css)
            cssLog = css.cssDataOutMsg.recorder()
            scSim.AddModelToTask(TASK_NAME, cssLog)
            cssLogs.append(cssLog)

    #   initialize Simulation
    #
    scSim.InitializeSimulation()
    #
    #   configure a simulation stop time and execute the simulation run
    #
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    #
    #   retrieve the logged data
    #
    if use_css_constellation:
        data_arrays.append((cssConstLog, cssConstLog.CosValue[:, : len(css_sensors)]))
    else:
        for css in cssLogs:
            data_arrays.append((css, css.OutputData))

    if is_archive:
        # store the data in an json archive
        x_data = []
        y_data = []
        if use_css_constellation:
            for idx in range(len(css_sensors)):
                x_data.append(data_arrays[0][0].times() * macros.NANO2SEC)
                y_data.append(data_arrays[0][1][:, idx])
        else:
            for index, data_css in enumerate(data_arrays):
                x_data.append(data_css[0].times() * macros.NANO2SEC)
                y_data.append(data_css[1])

        data = {
            "use_css_constellation": use_css_constellation,
            "use_platform": use_platform,
            "use_eclipse": use_eclipse,
            "use_kelly": use_kelly,
            "number_of_cycles": number_of_cycles,
            "number_of_sensors": number_of_sensors,
            "params": sensor_params,
            "x": np.array(x_data).tolist(),
            "y": np.array(y_data).tolist(),
        }
        json_data = dumps(data, indent=4)
        
        with open(f"{archive_name}", "w") as f:
            f.write(json_data)

    else:
        np.set_printoptions(precision=16)

        plot_results(data_arrays, css_sensors, use_css_constellation)


def plot_results(data_arrays, css_sensors, use_css_constellation):
    """Plot the results of the simulation using Plotly and save to specific directory."""
    fig = go.Figure()

    if use_css_constellation:
        for idx in range(len(css_sensors)):
            fig.add_trace(
                go.Scatter(
                    x=data_arrays[0][0].times() * macros.NANO2SEC,
                    y=data_arrays[0][1][:, idx],
                    mode="lines",
                    name=f"CSS_{idx}",
                )
            )
    else:
        for index, data_css in enumerate(data_arrays):
            fig.add_trace(
                go.Scatter(
                    x=data_css[0].times() * macros.NANO2SEC,
                    y=data_css[1],
                    mode="lines",
                    name=f"CSS_{index}",
                )
            )

    fig.update_layout(
        title="CSS Signals over Time",
        xaxis_title="Time [sec]",
        yaxis_title="CSS Signals",
        legend_title="Sensors",
    )

    fig.write_image(
        f"figs/css_signals_plot_N={len(css_sensors)}_use_css_constellation={use_css_constellation}.png"
    )


def setup_viz(scSim, task_name, scObject, css_sensors):
    """Setup visualization for the simulation."""
    viz = vizSupport.enableUnityVisualization(
        scSim, task_name, scObject, cssList=[css_sensors]
    )
    vizSupport.setInstrumentGuiSetting(
        viz,
        viewCSSPanel=True,
        viewCSSCoverage=True,
        viewCSSBoresight=True,
        showCSSLabels=True,
    )
    return viz


def create_simulation():
    """Create a simulation module as an empty container."""
    return SimulationBaseClass.SimBaseClass()


def set_simulation_time(sim_time_seconds):
    """Set the simulation time variable."""
    return macros.sec2nano(sim_time_seconds)


def create_simulation_process(scSim, task_name, time_step_seconds):
    """Create the simulation process and add a task to it."""
    dynProcess = scSim.CreateNewProcess(task_name)
    simulationTimeStep = macros.sec2nano(time_step_seconds)
    dynProcess.addTask(scSim.CreateNewTask(task_name, simulationTimeStep))
    return dynProcess


def setup_spacecraft():
    """Initialize spacecraft object and set properties."""
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "spacecraftBody"
    intertia = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [
        [0.0],
        [0.0],
        [0.0],
    ]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(intertia)

    # Set initial spacecraft states
    scObject.hub.r_CN_NInit = [[0.0], [0.0], [0.0]]  # m - r_CN_N
    scObject.hub.v_CN_NInit = [[0.0], [0.0], [0.0]]  # m/s - v_CN_N
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [
        [0.0],
        [0.0],
        [1.0 * macros.D2R],
    ]  # rad/s - omega_BN_B

    return scObject


def add_spacecraft_to_simulation(scSim, task_name, scObject):
    """Add spacecraft object to the simulation process."""
    scSim.AddModelToTask(task_name, scObject)


def create_sun_position_message():
    """Create a simulation message for the sun's position."""
    sunPositionMsgData = messaging.SpicePlanetStateMsgPayload()
    sunPositionMsgData.PositionVector = [0.0, om.AU * 1000.0, 0.0]
    return messaging.SpicePlanetStateMsg().write(sunPositionMsgData)


def create_eclipse_message():
    """Create a simulation message for eclipse conditions."""
    eclipseMsgData = messaging.EclipseMsgPayload()
    eclipseMsgData.shadowFactor = 0.5
    return messaging.EclipseMsg().write(eclipseMsgData)


if __name__ == "__main__":
    run_click()
