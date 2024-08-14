import inspect
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

filename = inspect.getframeinfo(inspect.currentframe()).filename
fileNameString = os.path.basename(os.path.splitext(__file__)[0])
path = os.path.dirname(os.path.abspath(filename))


from Basilisk import __path__
bskPath = __path__[0]


from Basilisk.architecture import messaging
from Basilisk.simulation import (coarseSunSensor, spacecraft)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion as om,
                                simIncludeGravBody, unitTestSupport)
from Basilisk.utilities.MonteCarlo.Controller import Controller, RetentionPolicy
from Basilisk.utilities.MonteCarlo.Dispersions import (UniformEulerAngleMRPDispersion, UniformDispersion,
                                                       NormalVectorCartDispersion, InertiaTensorDispersion)

NUMBER_OF_RUNS = 10
VERBOSE = True
MULTIPLE_INSTRUMENT = False
MULTIPLE_PLANET = False
SIM_TIME_STEP = 10.
NUM_DATA_POINTS = 500

# Here are the name of some messages that we want to retain or otherwise use
cssDataOutMsgName = ["css1DataOutMsg", "css2DataOutMsg", "css3DataOutMsg"]
scStateOutMsgName = "scStateOutMsg"


def runMonteCarlo():
    mcController = Controller()

    mcController.setSimulationFunction(createScenarioCSS)
    mcController.setExecutionFunction(executeScenario)
    mcController.setExecutionCount(NUMBER_OF_RUNS)
    mcController.setShouldDisperseSeeds(True)
    mcController.setShowProgressBar(True)
    # Optionally set the number of cores to use
    # mcController.setThreadCount(PROCESSES)
    mcController.setVerbose(VERBOSE)
    dirName = "montecarlo_test" + str(os.getpid())
    mcController.setArchiveDir(dirName)

    # Add the dispersions to the input parameters
    dispCSS1Fov = 'CSS1.fov'
    dispCSS2Fov = 'CSS2.fov'
    dispCSS3Fov = 'CSS3.fov'
    dispCSS1Orientation = 'CSS1.nHat_B'
    dispCSS2Orientation = 'CSS2.nHat_B'
    dispCSS3Orientation = 'CSS3.nHat_B'
    dispCSS1Position = 'CSS1.r_PB_B'
    dispCSS2Position = 'CSS2.r_PB_B'
    dispCSS3Position = 'CSS3.r_PB_B'
    dispInertia = 'hubref.IHubPntBc_B'
    dispList = {}# [dispCSS1Fov, dispCSS1Orientation]

    # Add dispersions with their dispersion type
    #mcController.addDispersion(InertiaTensorDispersion(dispInertia, stdAngle=0.1))
    mcController.addDispersion(NormalVectorCartDispersion(dispCSS1Orientation, [0., 0., 0.], [1., 1., 1.]))
    mcController.addDispersion(UniformDispersion(dispCSS1Fov, [(80. - 1. * 80.) * macros.D2R, (80. + 1.0 * 80.) * macros.D2R]))
    mcController.addDispersion(NormalVectorCartDispersion(dispCSS1Position, [0., 0., 0.], [1., 1., 1.]))
    if MULTIPLE_INSTRUMENT:
        mcController.addDispersion(NormalVectorCartDispersion(dispCSS2Orientation, [0., 0., 0.], [1., 1., 1.]))
        mcController.addDispersion(UniformDispersion(dispCSS2Fov, [(80. - 1. * 80.) * macros.D2R, (80. + 1.0 * 80.) * macros.D2R]))
        mcController.addDispersion(NormalVectorCartDispersion(dispCSS3Orientation, [0., 0., 0.], [1., 1., 1.]))
        mcController.addDispersion(UniformDispersion(dispCSS3Fov, [(80. - 1. * 80.) * macros.D2R, (80. + 1.0 * 80.) * macros.D2R]))
        mcController.addDispersion(NormalVectorCartDispersion(dispCSS2Position, [0., 0., 0.], [1., 1., 1.]))
        mcController.addDispersion(NormalVectorCartDispersion(dispCSS3Position, [0., 0., 0.], [1., 1., 1.]))

    # Add retention policy
    retentionPolicy = RetentionPolicy()
    retentionPolicy.addMessageLog(cssDataOutMsgName[0], ["OutputData"])
    if MULTIPLE_INSTRUMENT:
        retentionPolicy.addMessageLog(cssDataOutMsgName[1], ["OutputData"])
        retentionPolicy.addMessageLog(cssDataOutMsgName[2], ["OutputData"])
    retentionPolicy.addMessageLog(scStateOutMsgName, ["r_BN_N"])
    retentionPolicy.setDataCallback(plotSim)
    mcController.addRetentionPolicy(retentionPolicy)

    #InitialConditionRun(mcController)
    NormalMCRun(mcController, dirName, dispList)

    # Now we clean up data from this test
    shutil.rmtree(dirName)
    assert not os.path.exists(dirName), "No leftover data should exist after the test"

    # And possibly show the plots
    print("Test concluded, showing plots now via matplot...")
    plt.show()
    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")
    

def createScenarioCSS():  
    
    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"
    # Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()
    # Create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)
    # Create the dynamics task
    simulationTimeStep = macros.sec2nano(SIM_TIME_STEP)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))
    # Create sun message
    sunPositionMsg = messaging.SpicePlanetStateMsgPayload()
    sunPositionMsg.PositionVector = [-om.AU * 1000., 0.0, 0.0]
    sunMsg = messaging.SpicePlanetStateMsg().write(sunPositionMsg)

    # Create planet message (earth)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    # Create planet message (earth)
    planetCase1 = 'earth'
    planet1 = gravFactory.createEarth()
    planet1.isCentralBody = True  # ensure this is the central gravitational body
    req1 = planet1.radEquator

    planetPositionMsg1 = messaging.SpicePlanetStateMsgPayload()
    planetPositionMsg1.PositionVector = [0., 0., 0.]
    planetPositionMsg1.PlanetName = planetCase1
    planetPositionMsg1.J20002Pfix = np.identity(3)
    pl1Msg = messaging.SpicePlanetStateMsg().write(planetPositionMsg1)
    if MULTIPLE_PLANET:
        # Create planet message (moon)
        planetCase2 = 'moon'
        planetPositionMsg2 = messaging.SpicePlanetStateMsgPayload()
        planetPositionMsg2.PositionVector = [0., 384400. * 1000, 0.]
        planetPositionMsg2.PlanetName = planetCase2
        planetPositionMsg2.J20002Pfix = np.identity(3)
        pl2Msg = messaging.SpicePlanetStateMsg().write(planetPositionMsg2)

    #
    # Initialize spacecraft object and set properties
    #
    oe = om.ClassicElements()
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    rLEO = req1 + 500 * 1000  # m
    # Define the simulation inertia
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    scSim.hubref = scObject.hub
    if MULTIPLE_PLANET:
        # Set initial spacecraft states
        scObject.hub.r_CN_NInit = [[0.0], [rLEO], [0.0]]  # m - r_CN_N
        scObject.hub.v_CN_NInit = [[0.0], [0.0], [0.0]]  # m - v_CN_N
        scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # sigma_BN_B
        scObject.hub.omega_BN_BInit = [[0.0], [0.0], [1. * macros.D2R]]  # rad/s - omega_BN_B

    else:
        # Single planet case (earth)
        oe.a = rLEO
        oe.e = 0.0001
        oe.i = 0.0 * macros.D2R
        oe.Omega = 0.0 * macros.D2R
        oe.omega = 0.0 * macros.D2R
        oe.f = 180.0 * macros.D2R
        rN, vN = om.elem2rv(planet1.mu, oe)
        # set the simulation time
        n = np.sqrt(planet1.mu / oe.a / oe.a / oe.a)
        P = 2. * np.pi / n
        simulationTime = macros.sec2nano(0.5 * P)
        samplingTime = simulationTime // (NUM_DATA_POINTS-1)
        # Set initial spacecraft states
        scObject.hub.r_CN_NInit = rN  # m - r_CN_N
        scObject.hub.v_CN_NInit = vN  # m - v_CN_N
        scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # sigma_BN_B
        scObject.hub.omega_BN_BInit = [[0.0], [0.0], [.5 * macros.D2R]]  # rad/s - omega_BN_B
        gravFactory.addBodiesTo(scObject)

    # Add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    def setupCSS(CSS):
        CSS.stateInMsg.subscribeTo(scObject.scStateOutMsg)
        CSS.sunInMsg.subscribeTo(sunMsg)
        CSS.fov = 80. * macros.D2R
        CSS.maxOutput = 1.0
        CSS.nHat_B = np.array([1., 0., 0.])

    #
    # CSS-1
    #
    CSS1 = coarseSunSensor.CoarseSunSensor()
    CSS1.ModelTag = "CSS1"
    setupCSS(CSS1)

    if MULTIPLE_INSTRUMENT:
        # CSS-2
        CSS2 = coarseSunSensor.CoarseSunSensor()
        CSS2.ModelTag = "CSS2"
        setupCSS(CSS2)
        CSS2.nHat_B = np.array([-1., 0., 0.])
        # CSS-3
        CSS3 = coarseSunSensor.CoarseSunSensor()
        CSS3.ModelTag = "CSS3"
        setupCSS(CSS3)
        CSS3.nHat_B = np.array([0., -1., 0.])

    scSim.msgRecList = {}
    scSim.AddModelToTask(simTaskName, CSS1)
    scSim.CSS1 = CSS1
    scSim.msgRecList[cssDataOutMsgName[0]] = CSS1.cssDataOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, scSim.msgRecList[cssDataOutMsgName[0]])
    if MULTIPLE_INSTRUMENT:
        scSim.AddModelToTask(simTaskName, CSS2)
        scSim.CSS2 = CSS2
        scSim.AddModelToTask(simTaskName, CSS3)
        scSim.CSS3 = CSS3

    # setup logging
    scSim.msgRecList[scStateOutMsgName] = scObject.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, scSim.msgRecList[scStateOutMsgName])
    if MULTIPLE_INSTRUMENT:
        scSim.msgRecList[cssDataOutMsgName[1]] = CSS2.cssDataOutMsg.recorder(samplingTime)
        scSim.AddModelToTask(simTaskName, scSim.msgRecList[cssDataOutMsgName[1]])
        scSim.msgRecList[cssDataOutMsgName[2]] = CSS3.cssDataOutMsg.recorder(samplingTime)
        scSim.AddModelToTask(simTaskName, scSim.msgRecList[cssDataOutMsgName[2]])

    scSim.additionalReferences = [simulationTime, simulationTimeStep]

    return scSim

def executeScenario(sim):

    #   retrieve the module references
    simulationTime = sim.additionalReferences[0]

    #   initialize Simulation
    sim.InitializeSimulation()

    #   configure a simulation stop time and execute the simulation run
    sim.ConfigureStopTime(simulationTime)
    # Begin the simulation time run set above
    sim.ExecuteSimulation()

# This method is used to plot the retained data of a simulation.
# It is called once for each run of the simulation, overlapping the plots
def plotSim(data, retentionPolicy):
    #
    # Retrieve the logged data
    #
    n = 284
    #n = int(simulationTime / simulationTimeStep + 1)
    if MULTIPLE_INSTRUMENT:
        dataCSS = np.zeros(shape=(n, 3))
    else:
        dataCSS = np.zeros(shape=(n, 2))
    posData = data["messages"][scStateOutMsgName + ".r_BN_N"][:,1:]
    timeAxis = data["messages"][scStateOutMsgName + ".r_BN_N"][:,0]
    dataCSS[:, 0] = data["messages"][cssDataOutMsgName[0] + ".OutputData"][:,1]
    if MULTIPLE_INSTRUMENT:
        dataCSS[:, 1] = data["messages"][cssDataOutMsgName[1] + ".OutputData"][:,1]
        dataCSS[:, 2] = data["messages"][cssDataOutMsgName[2] + ".OutputData"][:,1]
    np.set_printoptions(precision=16)

    #
    # Plot the results
    #
    plt.figure(1)
    if MULTIPLE_INSTRUMENT:
        for idx in range(3):
            if not MULTIPLE_PLANET:
                plt.plot(timeAxis * macros.NANO2SEC, dataCSS[:, idx],
                        '--', linewidth=1.5, color=unitTestSupport.getLineColor(idx, 3),
                        label='Run ' + str(data["index"]) + ' - CSS$_' + str(idx) + '$')
    else:
        if not MULTIPLE_PLANET:
            plt.plot(timeAxis * macros.NANO2SEC, dataCSS[:, 0], 
                    label='Run ' + str(data["index"]) + ' - CSS$_{1}$')
    if MULTIPLE_PLANET:
        plt.legend(loc='upper center')
    else:
        plt.legend(loc='upper right')
    plt.xlabel('Time [s]')
    plt.ylabel('Instrument\'s signal')
    figureList = {}
    pltName = fileNameString + str(data["index"]) + str(1) + str(int(MULTIPLE_INSTRUMENT)) + str(
        int(MULTIPLE_PLANET))
    figureList[pltName] = plt.figure(1)
    if MULTIPLE_PLANET:
        # Show radius of SC
        plt.figure(2)
        fig = plt.gcf()
        ax = fig.gca()
        ax.ticklabel_format(useOffset=False, style='plain')
        rData = np.linalg.norm(posData, axis=1) / 1000.
        plt.plot(timeAxis * macros.NANO2SEC, rData, color='#aa0000')
        plt.xlabel('Time [s]')
        plt.ylabel('Radius [km]')
        pltName = fileNameString + str(data["index"]) + str(2) + str(int(MULTIPLE_INSTRUMENT)) + str(
            int(MULTIPLE_PLANET))
        figureList[pltName] = plt.figure(2)

    return figureList


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
    os.remove(icName + '/' + 'MonteCarlo.data')
    for i in range(numberICs):
        os.remove(icName + '/' + 'run' + str(i) + '.data')
    assert not os.path.exists(icName + '/' + 'MonteCarlo.data'), "No leftover data should exist after the test"


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
    retainedData = monteCarloLoaded.getRetainedData(NUMBER_OF_RUNS-1)
    assert retainedData is not None, "Retained data should be available after execution"
    assert "messages" in retainedData, "Retained data should retain messages"

    # We also can rerun a case using the same parameters and random seeds
    # If we rerun a properly set-up run, it should output the same data.
    # Here we test that if we rerun the case the data doesn't change
    oldOutput = retainedData["messages"][cssDataOutMsgName[0] + ".OutputData"]
 
    # Rerunning the case shouldn't fail
    failed = monteCarloLoaded.reRunCases([NUMBER_OF_RUNS-1])
    assert len(failed) == 0, "Should rerun case successfully"

    # Now access the newly retained data to see if it changed
    retainedData = monteCarloLoaded.getRetainedData(NUMBER_OF_RUNS-1)
    newOutput = retainedData["messages"][cssDataOutMsgName[0] + ".OutputData"]
    for k1, v1 in enumerate(oldOutput):
        for k2, v2 in enumerate(v1):
            assert math.fabs(oldOutput[k1][k2] - newOutput[k1][k2]) < .001, \
            "Outputs shouldn't change on runs if random seeds are same"

    # We can also access the initial parameters
    # The random seeds should differ between runs, so we will test that
    params1 = monteCarloLoaded.getParameters(NUMBER_OF_RUNS-1)
    params2 = monteCarloLoaded.getParameters(NUMBER_OF_RUNS-2)
    assert "TaskList[0].TaskModels[0].RNGSeed" in params1, "random number seed should be applied"
    for dispName in dispList:
        assert dispName in params1, "dispersion should be applied"
        # assert two different runs had different parameters.
        assert params1[dispName] != params2[dispName], "dispersion should be different in each run"

    # Now we execute our callback for the retained data.
    # For this run, that means executing the plot.
    # We can plot only runs 4,6,7 overlapped
    # monteCarloLoaded.executeCallbacks([4,6,7])
    # or execute the plot on all runs
    monteCarloLoaded.executeCallbacks([10, 50, 90])

if __name__ == "__main__":
    runMonteCarlo()

