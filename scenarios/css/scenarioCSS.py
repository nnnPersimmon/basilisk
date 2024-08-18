#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

r"""
Overview
--------

This script sets up a 6-DOF spacecraft in deep space without any gravitational bodies.
Only rotational  motion is simulated.  The script illustrates how to setup CSS
sensor units and log their data.  It is possible  to setup individual CSS sensors,
or setup a constellation or array of CSS sensors.

The script is found in the folder ``basilisk/examples`` and executed by using::

    python3 scenarioCSS.py

When the simulation completes a plot is shown for the CSS sensor signal history.

The simulation layout options (A) and (B) are shown in the following illustration.  A single simulation process is created which contains both the spacecraft simulation module, as well as two individual CSS sensor units.  In scenario (A) the CSS units are individually executed by the simulation, while scenario (B) uses a CSS constellation class that executes a list of CSS evaluations at the same time.

.. image:: /_images/static/test_scenarioCSS.svg
   :align: center

The dynamics simulation is setup using a :ref:`Spacecraft` module where a specific
spacecraft location is specified.  Note that both the rotational and translational
degrees of freedom of the spacecraft hub are turned on here to get a 6-DOF simulation.
The position  vector is required when computing the relative heading between the sun
and the spacecraft locations.  The  spacecraft position is held fixed, while the
orientation rotates constantly about the 3rd body axis.

The Field-Of-View variable fov must be specified.  This is the angle between the
sensor bore-sight and the edge of the field of view.  Beyond this angle all sensor
signals are set to zero. The scaleFactor variable scales a normalized CSS response
to this value if facing the sun head on.  The input message name InputSunMsg specifies
an input message that contains the sun's position. If sensor corruptions are to be
modeled, this can be set through the variables::

   CSS1.KellyFactor
   CSS1.SenBias
   CSS1.SenNoiseStd

The Kelly factor has values between 0 (off) and 1 and distorts the nominal cosine
response.  The SenBias  variable determines a normalized bias to be applied to the
CSS model, and SenNoiseStd provides Gaussian noise.

To create additional CSS sensor units, copies of the first CSS unit can be made.
This means only the parameters different in the additional units must be set.

A key parameter that remains is the CSS sensor unit normal vector.  There are
several options to set this vector (in body frame components).  The first
method is to set :math:`\hat{\mathbf n}` or ``nHat_B`` directly.  This is
done with::

   CSS1.nHat_B = np.array([1.0, 0.0, 0.0])
   CSS2.nHat_B = np.array([0.0, -1.0, 0.0])

Another option is to use a frame associated relative to a common CSS platform
:math:`\cal P`.  The bundled CSS units are often symmetrically arranged on a
platform such as in a pyramid configuration.  The the platform frame is  specified through::

   CSS1.setBodyToPlatformDCM(90.*macros.D2R, 0., 0.)

where the three orientation angles are 3-2-1 Euler angles.  These platform angles
are initialized to zero.  Next, the CSS unit direction vectors can be specified
through the azimuth and elevation angles (:math:`\phi`, :math:`\theta`).  These are (3)-(-2) Euler angles. ::

   CSS1.phi = 90.*macros.D2R
   CSS1.theta = 0.*macros.D2R

If no platform orientation is specified, then naturally these azimuth and elevation angles are
measured relative to the body frame :math:`\cal B`.

An optional input message is the solar eclipse message ``sunEclipseInMsg``.
If this message input name is specified for a CSS unit, then the eclipse
information is taken into account.  If this message name is not set, then
the CSS defaults to the spacecraft always being in the sun.

Illustration of Simulation Results
----------------------------------

The following images illustrate the expected simulation run returns for a range of script configurations.

::

    show_plots = True, useCSSConstellation=False, usePlatform=False, useEclipse=False, useKelly=False

This scenario simulates the CSS units being setup individually without any corruption.
The sensor unit normal axes are directly set, and no eclipse is modeled.
The signals of the two CSS units range from a maximum of 2 if the CSS axis is pointing
at the sun to zero.  The limited field of view of 80 degrees causes the sensor signal
to be clipped when the sun light incidence angle gets too small.

.. image:: /_images/Scenarios/scenarioCSS0000.svg
   :align: center

::

   show_plots = True, useCSSConstellation=False, usePlatform=True, useEclipse=False, useKelly=False

The resulting CSS sensor signals should be identical to the first scenario as the
chosen platform orientation and CSS azimuth and elevation angles are chosen to
yield the same senor normal unit axes.

.. image:: /_images/Scenarios/scenarioCSS0100.svg
   :align: center

::

   show_plots = True, useCSSConstellation=False, usePlatform=False, useEclipse=True, useKelly=False

The resulting CSS signals are scaled by a factor of 0.5 and are shown below.

.. image:: /_images/Scenarios/scenarioCSS0010.svg
  :align: center

::

    show_plots = True, useCSSConstellation=False, usePlatform=False, useEclipse=False, useKelly=True

This causes the CSS signals to become slightly warped, and depart from the nominal
cosine  behavior.

.. image:: /_images/Scenarios/scenarioCSS0001.svg
   :align: center

::

    show_plots = True, useCSSConstellation=True, usePlatform=False, useEclipse=False, useKelly=False

The resulting simulation results are shown below to be identical to the first setup as expected.

.. image:: /_images/Scenarios/scenarioCSS1000.svg
   :align: center

"""


#
# Basilisk Scenario Script and Integrated Test
#
# Purpose:  Demonstrates how to setup CSS sensors on a rigid spacecraft
# Author:   Hanspeter Schaub
# Creation Date:  July 21, 2017
#


import click
import numpy as np
import plotly.graph_objects as go

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__

# import message declarations
from Basilisk.architecture import messaging

# import simulation related support
from Basilisk.simulation import coarseSunSensor, spacecraft

# import general simulation support files
from Basilisk.utilities import (  # general support file with common unit test functions
    SimulationBaseClass,
    macros,
)
from Basilisk.utilities import orbitalMotion as om
from Basilisk.utilities import (  # general support file with common unit test functions
    unitTestSupport,
    vizSupport,
)


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
def run(
    use_css_constellation,
    use_platform,
    use_eclipse,
    use_kelly,
    number_of_cycles,
    number_of_sensors,
):
    """
    At the end of the python script you can specify the following example parameters.

    Args:
        useCSSConstellation (bool): Flag indicating if the CSS cluster/configuration class should be used.
        usePlatform (bool): Flag specifying if the CSS platform orientation should be set.
        useEclipse (bool): Flag indicating if the eclipse input message is used.
        useKelly (bool): Flag specifying if the Kelly corruption factor is used.

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
        CSS.fov = 80.0 * macros.D2R
        CSS.scaleFactor = 2.0
        CSS.maxOutput = 2.0
        CSS.minOutput = 0.5
        CSS.r_B = [2.00131, 2.36638, 1.0]
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
        if i >= 1:
            CSS.CSSGroupID = i - 1
            CSS.r_B = [-3.05, 0.55, 1.0]
        # Configure specific attributes for each sensor if needed
        if i == 1:
            CSS.CSSGroupID = 0
            CSS.r_B = [-3.05, 0.55, 1.0]
            if use_platform:
                CSS.theta = 0.0 * macros.D2R
                CSS.setUnitDirectionVectorWithPerturbation(0.0, 0.0)
            else:
                CSS.nHat_B = np.array([0.0, 1.0, 0.0])
        elif i == 2:
            CSS.CSSGroupID = 1
            CSS.fov = 45.0 * macros.D2R
            CSS.r_B = [-3.05, 0.55, 1.0]
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
    run()
