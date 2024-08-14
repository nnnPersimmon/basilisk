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

from config import (DEFAULT_THERMAL_SENSOR_CONFIG,
                    TAMPERED_IMPUT_SENSOR_CONFIGS,
                    TAMPERED_OUTPUT_SENSOR_CONFIGS)

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


def setup_thermal_sensor(spiceObject, scObject, scSim, config):
    # Now add the thermal sensor module
    thermalSensor = sensorThermal.SensorThermal()
    # Apply parameters from the configuration
    thermalSensor.T_0 = config.get(
        "T_0", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["T_0"]
    )
    thermalSensor.nHat_B = config.get(
        "nHat_B", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["nHat_B"]
    )
    thermalSensor.sensorArea = config.get(
        "sensorArea", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorArea"]
    )
    thermalSensor.sensorAbsorptivity = config.get(
        "sensorAbsorptivity",
        DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorAbsorptivity"],
    )
    thermalSensor.sensorEmissivity = config.get(
        "sensorEmissivity", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorEmissivity"]
    )
    thermalSensor.sensorMass = config.get(
        "sensorMass", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorMass"]
    )
    thermalSensor.sensorSpecificHeat = config.get(
        "sensorSpecificHeat",
        DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorSpecificHeat"],
    )
    thermalSensor.sensorPowerDraw = config.get(
        "sensorPowerDraw", DEFAULT_THERMAL_SENSOR_CONFIG["params"]["sensorPowerDraw"]
    )

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

    return scSim


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


def start_delay_simulation(scSim, locPoint, P, config):

    # Initialize the simulation
    scSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    scSim.ConfigureStopTime(
        macros.sec2nano(int(P + config["pre_switch_delay"]))
    )  # seconds to stop simulation

    # Begin the simulation time run set above
    scSim.ExecuteSimulation()

    # Change the location pointing vector and run the sim for another period
    locPoint.pHat_B = [0, 0, -1]
    scSim.ConfigureStopTime(
        macros.sec2nano(int(2 * (P + config["post_switch_delay"])))
    )  # seconds to stop simulation

    scSim.ExecuteSimulation()


def plot_results(tempLog, details):
    tempData = tempLog.temperature
    tvec = tempLog.times() * macros.NANO2HOUR

    plt.figure()
    plt.plot(tvec * 60.0, tempData)
    plt.xlabel("Time (min)")
    plt.ylabel("Temperature (deg C)")
    plt.grid(True)
    figureList = {"scenario_thermalSensor": plt.gcf()}
    plt.savefig(f"{details}.png")

    return figureList


def input_tampering():
    for config in [DEFAULT_THERMAL_SENSOR_CONFIG] + TAMPERED_IMPUT_SENSOR_CONFIGS:
        print(config)
        print(f"Running {config['description']}")
        scSim = setup_simulation()

        scObject = setup_spacecraft(scSim)

        gravFactory, mu = setup_grav_body(scObject)

        spiceObject = setup_spice_interface(gravFactory, scSim)

        rN, vN, n = setup_orbit(mu)

        mrpControl, locPoint, P = setup_modules(scSim, scObject, spiceObject, n, rN, vN)
        print(f"Setting up sensor with {config['params']}")
        tempLog = setup_thermal_sensor(spiceObject, scObject, scSim, config["params"])

        # Create the FSW vehicle configuration message
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        vehicleConfigOut.ISCPntB_B = intertia
        configDataMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
        mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)

        start_simulation(scSim, locPoint, P)

        plot_results(tempLog, details="input_" + config["description"])


def output_tampering():
    for config in TAMPERED_OUTPUT_SENSOR_CONFIGS:
        print(config)
        print(f"Running {config['description']}")
        scSim = setup_simulation()

        scObject = setup_spacecraft(scSim)

        gravFactory, mu = setup_grav_body(scObject)

        spiceObject = setup_spice_interface(gravFactory, scSim)

        rN, vN, n = setup_orbit(mu)

        mrpControl, locPoint, P = setup_modules(scSim, scObject, spiceObject, n, rN, vN)
        print(f"Setting up sensor with {config['params']}")
        tempLog = setup_thermal_sensor(spiceObject, scObject, scSim, config["params"])

        # Create the FSW vehicle configuration message
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        vehicleConfigOut.ISCPntB_B = intertia
        configDataMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
        mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)

        start_delay_simulation(scSim, locPoint, P, config["params"])
        plot_results(tempLog, details="output_" + config["description"])


################################################# INPUTS

# Initial Temperature (T_0): An attacker could manipulate the initial temperature to create false readings or disrupt the sensor's baseline functionality.
# Orientation (nHat_B): Changing the sensor's orientation could cause it to measure thermal radiation from unintended sources, leading to inaccurate data.
# Power Draw (sensorPowerDraw): Tampering with the power draw could disrupt the sensor's operation or cause it to overheat, leading to failure or incorrect readings.

########################### simulations
# Parameter Sensitivity Analysis

#     Range Testing: Define a range of plausible values for each parameter and systematically vary them within this range ; use better visualis.
#     Granularity: Test different granularities (e.g., small increments or decrements) to see how sensitive the system is to small changes in each parameter.
#     Extreme Values: Include both extreme values (e.g., very high or low) and non-physical values (e.g., negative area) to understand the boundaries and failure points.

#  Monte Carlo Simulation: Implement a Monte=Carlo simulation to randomly sample parameter values from predefined distributions. This can help in understanding the impact of random variations and their likelihood.

# Stress Testing

#     Load Testing: Simulate high-stress scenarios by pushing parameters to their limits (e.g., maximum power draw, extreme temperatures). >> we need to learn more on the sensor?? or assume the modeling??
#     Failure Modes: Test for failure modes by applying combinations of tampered parameters to see how the system behaves under stress.?? we need to get them from somewhere

########################### Countermeasures
# Validation and Verification

#     Cross-Verification: Implement cross-verification with redundant sensors or systems. Compare results from multiple sensors to detect anomalies.
#     Consistency Checks: Perform consistency checks on sensor inputs. For instance, verify that values fall within expected physical ranges and are consistent with other sensor data.


################################################# OUTPUT TIMING

# Pre-Switch Delay Impact: Delays before the orientation switch can affect how well the system stabilizes and prepares for the new orientation. If the system relies on certain conditions or initial settings being stabilized before the switch, a delay might either give it more time to adjust or cause issues if it's already nearing its limits.
# Post-Switch Delay Impact:  Delays after an orientation switch can influence how quickly and accurately the system adapts to the new conditions. The system's ability to quickly respond and stabilize in the new orientation is crucial for maintaining performance and avoiding errors.

if __name__ == "__main__":
    input_tampering()
    output_tampering()


# rerun for suntracker and attitude control

# check orientation

# use cases ;
# --thermal; fpga -> force active cooling or standby mode -> relate to azure
## add to the motivation
# --startracker;
# --attitude

# TODO:
# create the monte carlo for sensor thermal


# TODO:
# diff between noise and attacks; monte carlo


## if we do attack detection or countermeasures -> i get % performance and $ cost
# quantification of performance gain at expanse of other metrics (more comm reqs and more memory, cost etc)
# to reduce cost of monitoring, we can monitr every 10
# samples or 100 etc.. queue instead of per. so it gives param to tune the cost so it reduces mem and  much delay
# we need to have a benchmark; state of the art solution. this is the trivial way and this is what to suggest
# input parameters, associate ranges and why its needed

# countermeasure; add more sensors at the cost of higher payload mass and vol
