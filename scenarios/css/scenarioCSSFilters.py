import click
import numpy as np
from Basilisk import __path__

bskPath = __path__[0]

import matplotlib.pyplot as plt
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (okeefeEKF, sunlineEKF, sunlineSEKF,
                                    sunlineSuKF, sunlineUKF)
from Basilisk.simulation import coarseSunSensor, spacecraft
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.utilities import orbitalMotion as om
from Basilisk.utilities import unitTestSupport

import SunLineKF_test_utilities as Fplot


def setupUKFData(filterObject):
    """Setup UKF Filter Data"""
    filterObject.alpha = 0.02
    filterObject.beta = 2.0
    filterObject.kappa = 0.0

    filterObject.state = [1.0, 0.1, 0.0, 0.0, 0.01, 0.0]
    filterObject.covar = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
    ]
    qNoiseIn = np.identity(6)
    qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3] * 0.017 * 0.017
    qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6] * 0.0017 * 0.0017
    filterObject.qNoise = qNoiseIn.reshape(36).tolist()
    filterObject.qObsVal = 0.017**2
    filterObject.sensorUseThresh = np.sqrt(filterObject.qObsVal) * 5


def setupEKFData(filterObject):
    """Setup EKF Filter Data"""
    filterObject.state = [1.0, 0.1, 0.0, 0.0, 0.01, 0.0]
    filterObject.x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    filterObject.covar = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
    ]

    filterObject.qProcVal = 0.001**2
    filterObject.qObsVal = 0.017**2
    filterObject.sensorUseThresh = np.sqrt(filterObject.qObsVal) * 5

    filterObject.eKFSwitch = (
        5.0  # If low (0-5), the CKF kicks in easily, if high (>10) it's mostly only EKF
    )


def setupOEKFData(filterObject):
    """Setup OEKF Filter Data"""
    filterObject.omega = [0.0, 0.0, 0.0]
    filterObject.state = [1.0, 0.1, 0.0]
    filterObject.x = [0.0, 0.0, 0.0]
    filterObject.covar = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    filterObject.qProcVal = 0.1**2
    filterObject.qObsVal = 0.017**2
    filterObject.sensorUseThresh = np.sqrt(filterObject.qObsVal) * 5

    filterObject.eKFSwitch = (
        5.0  # If low (0-5), the CKF kicks in easily, if high (>10) it's mostly only EKF
    )


def setupSEKFData(filterObject):
    """Setup SEKF Filter Data"""
    filterObject.state = [1.0, 0.1, 0.0, 0.0, 0.0]
    filterObject.x = [0.0, 0.0, 0.0, 0.0, 0.0]
    filterObject.covar = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01,
    ]

    filterObject.qProcVal = 0.001**2
    filterObject.qObsVal = 0.017**2
    filterObject.sensorUseThresh = np.sqrt(filterObject.qObsVal) * 5

    filterObject.eKFSwitch = (
        5.0  # If low (0-5), the CKF kicks in easily, if high (>10) it's mostly only EKF
    )


def setupSuKFData(filterObject):
    """Setup SuKF Filter Data"""
    filterObject.alpha = 0.02
    filterObject.beta = 2.0
    filterObject.kappa = 0.0

    filterObject.stateInit = [1.0, 0.1, 0.0, 0.0, 0.0, 1.0]
    filterObject.covarInit = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0001,
    ]

    qNoiseIn = np.identity(6)
    qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3] * 0.001**2
    qNoiseIn[3:5, 3:5] = qNoiseIn[3:5, 3:5] * 0.0001**2
    qNoiseIn[5, 5] = qNoiseIn[5, 5] * 0.000001**2
    filterObject.qNoise = qNoiseIn.reshape(36).tolist()
    filterObject.qObsVal = 0.017**2
    filterObject.sensorUseThresh = np.sqrt(filterObject.qObsVal) * 5


@click.command()
@click.option(
    "--save-figures", is_flag=True, default=False, help="Save figures to file"
)
@click.option("--show-plots", is_flag=True, default=True, help="Show plots")
@click.option("--filter-type", default="SuKF", help="Filter type to use")
@click.option("--num-sensors", default=5, type=int, help="Number of sensors")
@click.option("--num-cycles", default=3, type=int, help="Number of cycles")
def run_click(save_figures, show_plots, filter_type, num_sensors, num_cycles):
    run(save_figures, show_plots, filter_type, num_sensors, num_cycles, 400)


def run(save_figures, show_plots, filter_type, num_sensors, num_cycles, simTime):
    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    # set the simulation time variable used later on
    simulationTime = macros.sec2nano(simTime * num_cycles)

    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(0.5)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #
    #   setup the simulation tasks/objects
    #
    # create sun position message at origin
    sunMsgData = messaging.SpicePlanetStateMsgPayload()
    sunMsg = messaging.SpicePlanetStateMsg().write(sunMsgData)
    sunLog = sunMsg.recorder()
    scSim.AddModelToTask(simTaskName, sunLog)

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    # define the simulation inertia
    I = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [
        [0.0],
        [0.0],
        [0.0],
    ]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    #
    # set initial spacecraft states
    #
    scObject.hub.r_CN_NInit = [[-om.AU * 1000.0], [0.0], [0.0]]  # m   - r_CN_N
    scObject.hub.v_CN_NInit = [[0.0], [0.0], [0.0]]  # m/s - v_CN_N
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [
        [-0.1 * macros.D2R],
        [0.5 * macros.D2R],
        [0.5 * macros.D2R],
    ]  # rad/s - omega_BN_B

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)
    dataLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, dataLog)

    # Make a CSS constelation
    cssConstelation = coarseSunSensor.CSSConstellation()
    CSSOrientationList = [
        [0.70710678118654746, -0.5, 0.5],
        [0.70710678118654746, -0.5, -0.5],
        [0.70710678118654746, 0.5, -0.5],
        [0.70710678118654746, 0.5, 0.5],
        [-0.70710678118654746, 0, 0.70710678118654757],
        [-0.70710678118654746, 0.70710678118654757, 0.0],
        [-0.70710678118654746, 0, -0.70710678118654757],
        [-0.70710678118654746, -0.70710678118654757, 0.0],
    ]

    def setupCSS(CSS):
        CSS.minOutput = 0.0
        CSS.senNoiseStd = 0.017
        CSS.sunInMsg.subscribeTo(sunMsg)
        CSS.stateInMsg.subscribeTo(scObject.scStateOutMsg)
        CSS.this.disown()

    for index in range(num_sensors):
        newCSS = coarseSunSensor.CoarseSunSensor()
        newCSS.ModelTag = f"CSS+{index}"
        setupCSS(newCSS)
        newCSS.nHat_B = CSSOrientationList[index]
        cssConstelation.appendCSS(newCSS)
    scSim.AddModelToTask(simTaskName, cssConstelation)

    #
    #   add the FSW CSS information
    #
    cssConstVehicle = messaging.CSSConfigMsgPayload()

    totalCSSList = []
    for index in range(num_sensors):
        newCSS = messaging.CSSUnitConfigMsgPayload()
        newCSS.nHat_B = CSSOrientationList[index]
        newCSS.CBias = 1.0
        totalCSSList.append(newCSS)
    cssConstVehicle.nCSS = num_sensors + 1
    cssConstVehicle.cssVals = totalCSSList

    cssConstMsg = messaging.CSSConfigMsg().write(cssConstVehicle)

    #
    # Setup filter
    #
    numStates = 6
    bVecLogger = None
    if filter_type == "EKF":
        module = sunlineEKF.sunlineEKF()
        module.ModelTag = "SunlineEKF"
        setupEKFData(module)

        # Add test module to runtime call list
        scSim.AddModelToTask(simTaskName, module)

    if filter_type == "OEKF":
        numStates = 3

        module = okeefeEKF.okeefeEKF()
        module.ModelTag = "okeefeEKF"
        setupOEKFData(module)

        # Add test module to runtime call list
        scSim.AddModelToTask(simTaskName, module)

    if filter_type == "uKF":
        module = sunlineUKF.sunlineUKF()
        module.ModelTag = "SunlineUKF"
        setupUKFData(module)

        # Add test module to runtime call list
        scSim.AddModelToTask(simTaskName, module)

    if filter_type == "SEKF":
        numStates = 5

        module = sunlineSEKF.sunlineSEKF()
        module.ModelTag = "SunlineSEKF"
        setupSEKFData(module)

        # Add test module to runtime call list
        scSim.AddModelToTask(simTaskName, module)
        bVecLogger = module.logger("bVec_B", simulationTimeStep)
        scSim.AddModelToTask(simTaskName, bVecLogger)

    if filter_type == "SuKF":
        numStates = 6
        module = sunlineSuKF.sunlineSuKF()
        module.ModelTag = "SunlineSuKF"
        setupSuKFData(module)

        # Add test module to runtime call list
        scSim.AddModelToTask(simTaskName, module)
        bVecLogger = module.logger("bVec_B", simulationTimeStep)
        scSim.AddModelToTask(simTaskName, bVecLogger)

    module.cssDataInMsg.subscribeTo(cssConstelation.constellationOutMsg)
    module.cssConfigInMsg.subscribeTo(cssConstMsg)

    navLog = module.navStateOutMsg.recorder()
    filtLog = module.filtDataOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, navLog)
    scSim.AddModelToTask(simTaskName, filtLog)

    #
    #   initialize Simulation
    #
    scSim.InitializeSimulation()

    #
    #   configure a simulation stop time and execute the simulation run
    #
    scSim.ConfigureStopTime(simulationTime)

    # Time the runs for performance comparisons
    scSim.ExecuteSimulation()

    #
    #   retrieve the logged data
    #
    def addTimeColumn(time, data):
        return np.transpose(np.vstack([[time], np.transpose(data)]))

    # Get messages that will make true data
    timeAxis = dataLog.times()
    OutSunPos = addTimeColumn(timeAxis, sunLog.PositionVector)
    Outr_BN_N = addTimeColumn(timeAxis, dataLog.r_BN_N)
    OutSigma_BN = addTimeColumn(timeAxis, dataLog.sigma_BN)
    Outomega_BN = addTimeColumn(timeAxis, dataLog.omega_BN_B)

    # Get the filter outputs through the messages
    stateLog = addTimeColumn(timeAxis, filtLog.state[:, range(numStates)])
    postFitLog = addTimeColumn(timeAxis, filtLog.postFitRes[:, :8])
    covarLog = addTimeColumn(timeAxis, filtLog.covar[:, range(numStates * numStates)])
    obsLog = addTimeColumn(timeAxis, filtLog.numObs)

    # Get bVec_B through the variable logger
    bVecLog = None if bVecLogger is None else addTimeColumn(timeAxis, bVecLogger.bVec_B)

    dcmLog = np.zeros([len(stateLog[:, 0]), 3, 3])
    omegaExp = np.zeros([len(stateLog[:, 0]), 3])
    if filter_type == "SEKF":
        dcm = sunlineSEKF.new_doubleArray(3 * 3)
        for j in range(9):
            sunlineSEKF.doubleArray_setitem(dcm, j, 0)
        for i in range(len(stateLog[:, 0])):
            sunlineSEKF.sunlineSEKFComputeDCM_BS(
                stateLog[i, 1:4].tolist(), bVecLog[i, 1:4].tolist(), dcm
            )
            dcmOut = []
            for j in range(9):
                dcmOut.append(sunlineSEKF.doubleArray_getitem(dcm, j))
            dcmLog[i, :, :] = np.array(dcmOut).reshape([3, 3])
            omegaExp[i, :] = -np.dot(
                dcmLog[i, :, :], np.array([0, stateLog[i, 4], stateLog[i, 5]])
            )
    if filter_type == "SuKF":
        dcm = sunlineSuKF.new_doubleArray(3 * 3)
        for j in range(9):
            sunlineSuKF.doubleArray_setitem(dcm, j, 0)
        for i in range(len(stateLog[:, 0])):
            sunlineSuKF.sunlineSuKFComputeDCM_BS(
                stateLog[i, 1:4].tolist(), bVecLog[i, 1:4].tolist(), dcm
            )
            dcmOut = []
            for j in range(9):
                dcmOut.append(sunlineSuKF.doubleArray_getitem(dcm, j))
            dcmLog[i, :, :] = np.array(dcmOut).reshape([3, 3])
            omegaExp[i, :] = np.dot(dcmLog[i, :, :].T, Outomega_BN[i, 1:])

    sHat_B = np.zeros(np.shape(OutSunPos))
    sHatDot_B = np.zeros(np.shape(OutSunPos))
    for i in range(len(OutSunPos[:, 0])):
        sHat_N = (OutSunPos[i, 1:] - Outr_BN_N[i, 1:]) / np.linalg.norm(
            OutSunPos[i, 1:] - Outr_BN_N[i, 1:]
        )
        dcm_BN = rbk.MRP2C(OutSigma_BN[i, 1:])
        sHat_B[i, 0] = sHatDot_B[i, 0] = OutSunPos[i, 0]
        sHat_B[i, 1:] = np.dot(dcm_BN, sHat_N)
        sHatDot_B[i, 1:] = -np.cross(Outomega_BN[i, 1:], sHat_B[i, 1:])

    expected = np.zeros(np.shape(stateLog))
    expected[:, 0:4] = sHat_B
    # The OEKF has fewer states
    if filter_type != "OEKF" and filter_type != "SEKF" and filter_type != "SuKF":
        expected[:, 4:] = sHatDot_B[:, 1:]
    if filter_type == "SEKF" or filter_type == "SuKF":
        for i in range(len(stateLog[:, 0])):
            expected[i, 4] = omegaExp[i, 1]
            expected[i, 5] = omegaExp[i, 2]

    #   plot the results
    #
    errorVsTruth = np.copy(stateLog)
    errorVsTruth[:, 1:] -= expected[:, 1:]

    Fplot.StateErrorCovarPlot(
        errorVsTruth, covarLog, filter_type, show_plots, save_figures
    )
    Fplot.StatesVsExpected(
        stateLog, covarLog, expected, filter_type, show_plots, save_figures
    )
    Fplot.PostFitResiduals(
        num_sensors,
        postFitLog,
        np.sqrt(module.qObsVal),
        filter_type,
        show_plots,
        save_figures,
    )
    Fplot.numMeasurements(obsLog, filter_type, show_plots, save_figures)

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return


if __name__ == "__main__":
    run_click()
