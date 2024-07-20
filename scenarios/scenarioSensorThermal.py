import os

import matplotlib.pyplot as plt
import numpy as np
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import locationPointing, mrpFeedback
from Basilisk.simulation import (ephemerisConverter, extForceTorque,
                                 sensorThermal, simpleNav, spacecraft)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, unitTestSupport)

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


simTaskName = "simTask"
simProcessName = "simProcess"


intertia = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]


def setup_spacecraft(scSim):
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(intertia)
    scSim.AddModelToTask(simTaskName, scObject)
    return scObject


def setup_orbit(mu):
    oe = orbitalMotion.ClassicElements()
    oe.a = (6378 + 600) * 1000.0
    oe.e = 0.01
    oe.i = 63.3 * macros.D2R
    oe.Omega = 88.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 135.3 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    return rN, vN, np.sqrt(mu / oe.a / oe.a / oe.a)


def setup_thermal_sensor(spiceObject, scObject, scSim):
    # Now add the thermal sensor module
    thermalSensor = sensorThermal.SensorThermal()
    thermalSensor.T_0 = 0  # Celsius
    thermalSensor.nHat_B = [0, 0, 1]
    thermalSensor.sensorArea = 1.0  # m^2
    thermalSensor.sensorAbsorptivity = 0.25
    thermalSensor.sensorEmissivity = 0.34
    thermalSensor.sensorMass = 2.0  # kg
    thermalSensor.sensorSpecificHeat = 890
    thermalSensor.sensorPowerDraw = 30.0  # W
    thermalSensor.sunInMsg.subscribeTo(spiceObject.planetStateOutMsgs[0])
    thermalSensor.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    scSim.AddModelToTask(simTaskName, thermalSensor)
    tempLog = thermalSensor.temperatureOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, tempLog)

    return tempLog


def setup_spice_interface(gravFactory, scSim):
    spiceObject = gravFactory.createSpiceInterface(
        time="2021 MAY 04 07:47:48.965 (UTC)"
    )
    scSim.AddModelToTask(simTaskName, spiceObject)

    return spiceObject


def setup_grav_body(scObject):
    # Create the gravFactory
    gravFactory = simIncludeGravBody.gravBodyFactory()

    # Create the sun
    gravFactory.createSun()

    # Set up Earth Gravity Body
    earth = gravFactory.createEarth()
    earth.isCentralBody = True  # ensure this is the central gravitational body
    mu = earth.mu

    # Attach gravity model to spacecraft
    gravFactory.addBodiesTo(scObject)

    return gravFactory, mu


def setup_modules(scSim, scObject, spiceObject, n, rN, vN):
    P = 2.0 * np.pi / n

    scObject.hub.r_CN_NInit = rN  # m   - r_CN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_CN_N
    scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]  # rad/s - omega_BN_B

    # Set up extForceTorque module
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "externalDisturbance"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject, 97)

    # Add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject, ModelPriority=101)
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Create the ephemeris converter
    ephemConverter = ephemerisConverter.EphemerisConverter()
    ephemConverter.ModelTag = "ephemConverter"
    ephemConverter.addSpiceInputMsg(spiceObject.planetStateOutMsgs[0])
    ephemConverter.addSpiceInputMsg(spiceObject.planetStateOutMsgs[1])
    scSim.AddModelToTask(simTaskName, ephemConverter)

    # Set up sun pointing guidance module
    locPoint = locationPointing.locationPointing()
    locPoint.ModelTag = "locPoint"
    scSim.AddModelToTask(simTaskName, locPoint, 99)
    locPoint.pHat_B = [0, 0, 1]
    locPoint.scAttInMsg.subscribeTo(sNavObject.attOutMsg)
    locPoint.scTransInMsg.subscribeTo(sNavObject.transOutMsg)
    locPoint.celBodyInMsg.subscribeTo(ephemConverter.ephemOutMsgs[0])

    # Set up the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    scSim.AddModelToTask(simTaskName, mrpControl, 98)
    mrpControl.guidInMsg.subscribeTo(locPoint.attGuidOutMsg)
    mrpControl.K = 5.5
    mrpControl.Ki = -1  # make value negative to turn off integral feedback
    mrpControl.P = 30.0
    mrpControl.integralLimit = 2.0 / mrpControl.Ki * 0.1

    # Connect torque command to external torque effector
    extFTObject.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    return mrpControl, locPoint, P


def setup_simulation():
    scSim = SimulationBaseClass.SimBaseClass()
    simulationTimeStep = macros.sec2nano(1.0)

    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    return scSim, simTaskName


def start_simulation(scSim, locPoint, P):
    # Initialize the simulation
    scSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    scSim.ConfigureStopTime(macros.sec2nano(int(P)))  # seconds to stop simulation

    # Begin the simulation time run set above
    scSim.ExecuteSimulation()

    # Change the location pointing vector and run the sim for another period
    locPoint.pHat_B = [0, 0, -1]
    scSim.ConfigureStopTime(macros.sec2nano(int(2 * P)))  # seconds to stop simulation
    scSim.ExecuteSimulation()


def plot_results(tempLog):
    tempData = tempLog.temperature
    tvec = tempLog.times() * macros.NANO2HOUR

    figureList = {}
    plt.close("all")
    plt.figure(1)
    plt.plot(tvec * 60.0, tempData)
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (deg C)")
    plt.grid(True)
    figureList["scenario_thermalSensor"] = plt.figure(1)

    plt.show()
    plt.close("all")


def run():
    scSim, simTaskName = setup_simulation()

    scObject = setup_spacecraft(scSim)

    gravFactory, mu = setup_grav_body(scObject)

    spiceObject = setup_spice_interface(gravFactory, scSim)

    rN, vN, n = setup_orbit(mu)

    mrpControl, locPoint, P = setup_modules(scSim, scObject, spiceObject, n, rN, vN)
    tempLog = setup_thermal_sensor(spiceObject, scObject, scSim)

    # Create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = intertia
    configDataMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
    mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)

    start_simulation(scSim, locPoint, P)

    plot_results(tempLog)


if __name__ == "__main__":
    run()
