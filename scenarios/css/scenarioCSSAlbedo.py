import os

import click
import matplotlib.pyplot as plt
import numpy as np
# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
# import message declarations
from Basilisk.architecture import messaging
# import simulation related support
from Basilisk.simulation import albedo, coarseSunSensor, eclipse, spacecraft
# import general simulation support files
from Basilisk.utilities import (  # general support file with common unit test functions
    SimulationBaseClass, macros)
from Basilisk.utilities import orbitalMotion as om
from Basilisk.utilities import (  # general support file with common unit test functions
    simIncludeGravBody, unitTestSupport)

from config import DEFAULT_CSS_CONFIG

bskPath = __path__[0]
fileNameString = os.path.basename(os.path.splitext(__file__)[0])


def create_simulation():
    """Create and return the simulation base class instance and process."""
    simTaskName = TASK_NAME
    simProcessName = TASK_NAME
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(10.0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))
    return scSim, simTaskName, simulationTimeStep


def create_sun_message():
    """Create and return the sun position message."""
    sunPositionMsg = messaging.SpicePlanetStateMsgPayload()
    sunPositionMsg.PositionVector = [-om.AU * 1000.0, 0.0, 0.0]
    return messaging.SpicePlanetStateMsg().write(sunPositionMsg)


def create_planet_messages(multiplePlanet):
    """Create and return planet messages."""
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planetMessages = {}

    # Create planet message (earth)
    planetCase1 = "earth"
    planet1 = gravFactory.createEarth()
    planet1.isCentralBody = True
    req1 = planet1.radEquator
    planetPositionMsg1 = messaging.SpicePlanetStateMsgPayload()
    planetPositionMsg1.PositionVector = [0.0, 0.0, 0.0]
    planetPositionMsg1.PlanetName = planetCase1
    planetPositionMsg1.J20002Pfix = np.identity(3)
    planetMessages["earth"] = messaging.SpicePlanetStateMsg().write(planetPositionMsg1)

    if multiplePlanet:
        # Create planet message (moon)
        planetCase2 = "moon"
        planetPositionMsg2 = messaging.SpicePlanetStateMsgPayload()
        planetPositionMsg2.PositionVector = [0.0, 384400.0 * 1000, 0.0]
        planetPositionMsg2.PlanetName = planetCase2
        planetPositionMsg2.J20002Pfix = np.identity(3)
        planetMessages["moon"] = messaging.SpicePlanetStateMsg().write(
            planetPositionMsg2
        )

    return planetMessages, req1, planet1, gravFactory


def initialize_spacecraft(multiplePlanet, req, planet, gravFactory):
    """Initialize and return the spacecraft object."""
    oe = om.ClassicElements()
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    rLEO = req + 500 * 1000  # m

    # Define the simulation inertia
    I = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]
    scObject.hub.mHub = 750.0  # kg
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    if multiplePlanet:
        scObject.hub.r_CN_NInit = [[0.0], [rLEO], [0.0]]
        scObject.hub.v_CN_NInit = [[0.0], [0.0], [0.0]]
        scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
        scObject.hub.omega_BN_BInit = [[0.0], [0.0], [1.0 * macros.D2R]]
    else:
        oe.a = rLEO
        oe.e = 0.0001
        oe.i = 0.0 * macros.D2R
        oe.Omega = 0.0 * macros.D2R
        oe.omega = 0.0 * macros.D2R
        oe.f = 180.0 * macros.D2R
        rN, vN = om.elem2rv(planet.mu, oe)
        n = np.sqrt(planet.mu / oe.a / oe.a / oe.a)
        P = 2.0 * np.pi / n
        simulationTime = macros.sec2nano(0.5 * P)
        scObject.hub.r_CN_NInit = rN
        scObject.hub.v_CN_NInit = vN
        scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
        scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.5 * macros.D2R]]
        gravFactory.addBodiesTo(scObject)

    return scObject, simulationTime


TASK_NAME = "css_albedo_simulation"


def setup_albedo(scObject, sunMsg, use_eclipse, message):
    albModule = albedo.Albedo()
    albModule.ModelTag = "AlbedoModule"
    albModule.spacecraftStateInMsg.subscribeTo(scObject.scStateOutMsg)
    albModule.sunPositionInMsg.subscribeTo(sunMsg)

    if use_eclipse:
        albModule.eclipseCase = True
        eclipseObject = eclipse.Eclipse()
        eclipseObject.sunInMsg.subscribeTo(sunMsg)
        eclipseObject.addSpacecraftToModel(scObject.scStateOutMsg)
        eclipseObject.addPlanetToModel(message)
        return eclipseObject, albModule
    return None, albModule


@click.command()
@click.option("--show-plots", is_flag=True, default=True, help="Show plots")
@click.option("--albedo-data", is_flag=False, default=False, help="Use albedo data")
@click.option(
    "--multiple-instrument",
    is_flag=True,
    default=False,
    help="Use multiple instruments",
)
@click.option(
    "--multiple-planet", is_flag=True, default=False, help="Use multiple planets"
)
@click.option("--use-eclipse", is_flag=True, default=True, help="Use eclipse")
@click.option("--num-cycles", type=int, default=1, help="Number of cycles")
@click.option("--num-sensors", type=int, default=1, help="Number of sensors")
def run_click(
    show_plots,
    albedo_data,
    multiple_instrument,
    multiple_planet,
    use_eclipse,
    num_cycles,
    num_sensors,
):
    run(
        show_plots,
        albedo_data,
        multiple_instrument,
        multiple_planet,
        use_eclipse,
        num_cycles,
        num_sensors,
    )


def run(
    show_plots,
    albedoData,
    multipleInstrument,
    multiplePlanet,
    use_eclipse,
    num_cycles,
    number_of_sensors,
):
    print(number_of_sensors)
    scSim, simTaskName, simulationTimeStep = create_simulation()
    sunMsg = create_sun_message()
    planetMessages, req, planet, gravFactory = create_planet_messages(multiplePlanet)
    scObject, simulationTime = initialize_spacecraft(
        multiplePlanet, req, planet, gravFactory
    )

    # Add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    eclipseObject, albModule = setup_albedo(
        scObject, sunMsg, use_eclipse, planetMessages["earth"]
    )
    scSim.AddModelToTask(simTaskName, eclipseObject)

    if albedoData:
        dataPath = os.path.abspath(bskPath + "/supportData/AlbedoData/")
        fileName = "Earth_ALB_2018_CERES_All_5x5.csv"
        albModule.addPlanetandAlbedoDataModel(
            planetMessages["earth"], dataPath, fileName
        )
    else:
        ALB_avg, numLat, numLon = 0.5, 200, 200
        albModule.addPlanetandAlbedoAverageModel(
            planetMessages["earth"], ALB_avg, numLat, numLon
        )
    #
    if multiplePlanet:
        albModule.addPlanetandAlbedoAverageModel(planetMessages["moon"])

    def setupCSS(CSS):
        CSS.stateInMsg.subscribeTo(scObject.scStateOutMsg)
        CSS.sunInMsg.subscribeTo(sunMsg)
        CSS.fov = 80.0 * macros.D2R
        CSS.maxOutput = 1.0
        CSS.nHat_B = np.array([1.0, 0.0, 0.0])
        if use_eclipse:
            CSS.sunEclipseInMsg.subscribeTo(eclipseObject.eclipseOutMsgs[0])

    css_sensors = []
    cssLogs = []
    data_arrays = []
    alb_data_arrays = []

    for i in range(number_of_sensors):
        CSS = coarseSunSensor.CoarseSunSensor()
        CSS.ModelTag = f"CSS{i+1}_sensor"
        setupCSS(CSS)
        #
        # Add instrument to albedo module
        #
        config = albedo.instConfig_t()
        config.fov = CSS.fov
        config.nHat_B = CSS.nHat_B
        config.r_IB_B = CSS.r_PB_B
        albModule.addInstrumentConfig(config)
        # CSS albedo input message names should be defined after adding instrument to module
        CSS.albedoInMsg.subscribeTo(albModule.albOutMsgs[i])

    #
    # Add albedo and CSS to task and setup logging before the simulation is initialized
    #
    scSim.AddModelToTask(simTaskName, albModule)

    for css in css_sensors:
        scSim.AddModelToTask(TASK_NAME, css)
        cssLog = css.cssDataOutMsg.recorder()
        scSim.AddModelToTask(TASK_NAME, cssLog)
        cssLogs.append(cssLog)

    # setup logging
    dataLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, dataLog)

    for i in range(number_of_sensors):
        albLog = albModule.albOutMsgs[i].recorder()
        scSim.AddModelToTask(simTaskName, albLog)
        alb_data_arrays.append(albLog.albedoAtInstrument)

    for css in cssLogs:
        data_arrays.append((css, css.OutputData))

    #
    # Initialize Simulation
    #
    scSim.InitializeSimulation()
    #
    if multiplePlanet:
        velRef = scObject.dynManager.getStateObject("hubVelocity")
        # Configure a simulation stop time and execute the simulation run
        T1 = macros.sec2nano(500.0)
        scSim.ConfigureStopTime(T1)
        scSim.ExecuteSimulation()
        # get the current spacecraft states
        vVt = unitTestSupport.EigenVector3d2np(velRef.getState())
        T2 = macros.sec2nano(1000.0)
        # Set second spacecraft states for decrease in altitude
        vVt = vVt + [0.0, 375300, 0.0]  # m - v_CN_N
        velRef.setState(vVt)
        scSim.ConfigureStopTime(T1 + T2)
        scSim.ExecuteSimulation()
        # get the current spacecraft states
        T3 = macros.sec2nano(500.0)
        # Set second spacecraft states for decrease in altitude
        vVt = [0.0, 0.0, 0.0]  # m - v_CN_N
        velRef.setState(vVt)
        scSim.ConfigureStopTime(T1 + T2 + T3)
        scSim.ExecuteSimulation()
        simulationTime = T1 + T2 + T3
    else:
        # Configure a simulation stop time and execute the simulation run
        scSim.ConfigureStopTime(simulationTime)
        scSim.ExecuteSimulation()
    #
    # Retrieve the logged data
    #
    n = int(simulationTime / simulationTimeStep + 1)
    if multipleInstrument:
        dataCSS = np.zeros(shape=(n, 3))
        dataAlb = np.zeros(shape=(n, 3))
    else:
        dataCSS = np.zeros(shape=(n, 2))
        dataAlb = np.zeros(shape=(n, 2))
    posData = dataLog.r_BN_N

    # dataCSS[:, 0] = css1Log.OutputData
    # dataAlb[:, 0] = alb0Log.albedoAtInstrument
    # if multipleInstrument:
    #     dataCSS[:, 1] = css2Log.OutputData
    #     dataCSS[:, 2] = css3Log.OutputData
    #     dataAlb[:, 1] = alb1Log.albedoAtInstrument
    #     dataAlb[:, 2] = alb2Log.albedoAtInstrument
    np.set_printoptions(precision=16)
    #
    # Plot the results
    #
    plt.close("all")  # clears out plots from earlier test runs
    plt.figure(1)
    timeAxis = dataLog.times()
    if multipleInstrument:
        for idx in range(number_of_sensors):
            plt.plot(
                timeAxis * macros.NANO2SEC,
                alb_data_arrays[:, idx],
                linewidth=2,
                alpha=0.7,
                color=unitTestSupport.getLineColor(idx, 3),
                label="Albedo$_{" + str(idx) + "}$",
            )
            if not multiplePlanet:
                plt.plot(
                    timeAxis * macros.NANO2SEC,
                    data_arrays[:, idx],
                    "--",
                    linewidth=1.5,
                    color=unitTestSupport.getLineColor(idx, 3),
                    label="CSS$_{" + str(idx) + "}$",
                )
    else:
        plt.plot(
            timeAxis * macros.NANO2SEC,
            dataAlb,
            linewidth=2,
            alpha=0.7,
            color=unitTestSupport.getLineColor(0, 2),
            label="Alb$_{1}$",
        )
        if not multiplePlanet:
            plt.plot(
                timeAxis * macros.NANO2SEC,
                dataCSS,
                "--",
                linewidth=1.5,
                color=unitTestSupport.getLineColor(1, 2),
                label="CSS$_{1}$",
            )
    if multiplePlanet:
        plt.legend(loc="upper center")
    else:
        plt.legend(loc="upper right")
    plt.xlabel("Time [s]")
    plt.ylabel("Instrument's signal")
    figureList = {}
    pltName = (
        fileNameString
        + str(1)
        + str(int(albedoData))
        + str(int(multipleInstrument))
        + str(int(multiplePlanet))
        + str(int(use_eclipse))
    )
    figureList[pltName] = plt.figure(1)
    if multiplePlanet:
        # Show radius of SC
        plt.figure(2)
        fig = plt.gcf()
        ax = fig.gca()
        ax.ticklabel_format(useOffset=False, style="plain")
        rData = np.linalg.norm(posData, axis=1) / 1000.0
        plt.plot(timeAxis * macros.NANO2SEC, rData, color="#aa0000")
        plt.xlabel("Time [s]")
        plt.ylabel("Radius [km]")
        pltName = (
            fileNameString
            + str(2)
            + str(int(albedoData))
            + str(int(multipleInstrument))
            + str(int(multiplePlanet))
            + str(int(use_eclipse))
        )
        figureList[pltName] = plt.figure(2)

    if albedoData:
        filePath = os.path.abspath(dataPath + "/" + fileName)
        ALB1 = np.genfromtxt(filePath, delimiter=",")
        # ALB coefficient figures
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.set_title("Earth Albedo Coefficients (All Sky)")
        ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)")
        plt.imshow(ALB1, cmap="Reds", interpolation="none", extent=[-180, 180, 90, -90])
        plt.colorbar(orientation="vertical")
        ax.set_ylim(ax.get_ylim()[::-1])
        pltName = (
            fileNameString
            + str(2)
            + str(int(albedoData))
            + str(int(multipleInstrument))
            + str(int(multiplePlanet))
            + str(int(use_eclipse))
        )
        figureList[pltName] = plt.figure(2)

    if show_plots:
        plt.show()
    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")
    return figureList


if __name__ == "__main__":
    run_click()
