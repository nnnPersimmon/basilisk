
# CSS Simulations

## Overview

This script is a modification of the original Basilisk scenario script created by Hanspeter Schaub on July 21, 2017. The purpose of the script is to demonstrate how to set up Coarse Sun Sensor (CSS) units on a rigid spacecraft. The original script has been extended to include additional simulation options and configurations and input tampering with Monte Carlo.

## Purpose

The original script was designed to set up a 6-DOF spacecraft in deep space, simulating rotational motion without any gravitational bodies. It illustrates how to configure CSS sensor units and log their data. The script allows for setting up individual CSS sensors or configuring a constellation or array of CSS sensors.

## Script Location

The script can be found in the folder `basilisk/examples` and is executed using the following command:

```bash
python3 scenarioCSS.py [OPTIONS]
```

When the simulation completes, a plot showing the CSS sensor signal history is displayed.

## Command-Line Parameters

The following options can be used to customize the simulation:

- `--use-css-constellation`: Use the CSS Constellation class to evaluate a list of CSS sensors simultaneously (default: `False`).
- `--use-platform`: Use a specified platform orientation for the CSS sensors (default: `False`).
- `--use-eclipse`: Account for solar eclipses in the simulation (default: `False`).
- `--use-kelly`: Apply the Kelly corruption factor to the sensor data (default: `False`).
- `--number-of-cycles`: Specify the number of simulation cycles (default: `5`, must be `1` or more).
- `--number-of-sensors`: Specify the number of CSS sensors to simulate (default: `3`, must be `1` or more).

### Example Usage

```bash
python3 scenarioCSS.py --use-css-constellation --number-of-cycles 10 --number-of-sensors 5
```

In this example, the simulation will run with the CSS Constellation class enabled, for 10 cycles, and with 5 CSS sensors.

## Setup and Example Run

To set up the environment and run the simulation, follow these steps:

1. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

2. Navigate to the CSS scenario directory:

    ```bash
    cd scenarios/css
    ```

3. Run the simulation with your desired options:

    ```bash
    python scenarioCSS.py --use-css-constellation
    ```

## Dynamics Simulation Setup

The dynamics simulation is set up using a Spacecraft module, where a specific spacecraft location is specified. The simulation is a 6-DOF setup, with both rotational and translational degrees of freedom activated. The spacecraft position is held fixed, while its orientation rotates constantly about the third body axis.

## Key Parameters and Configurations

- **Field-Of-View (FOV):** Specifies the angle between the sensor bore-sight and the edge of the field of view. Beyond this angle, all sensor signals are set to zero.
- **Scale Factor:** Scales a normalized CSS response to this value when facing the sun directly.
- **InputSunMsg:** Specifies an input message containing the sun's position.
- **Kelly Factor:** Distorts the nominal cosine response, with values between 0 (off) and 1.
- **Sensor Bias (SenBias):** Applies a normalized bias to the CSS model.
- **Sensor Noise Standard Deviation (SenNoiseStd):** Adds Gaussian noise to the sensor signal.

### CSS Sensor Unit Configuration

To create additional CSS sensor units, copies of the first CSS unit can be made, with only the differing parameters needing to be set.

- **Normal Vector:** The CSS sensor unit normal vector can be set directly in body frame components or relative to a common CSS platform frame.
- **Platform Frame:** A platform frame can be specified using 3-2-1 Euler angles. Azimuth and elevation angles can also be set relative to this platform.

Example:

```python
CSS1.nHat_B = np.array([1.0, 0.0, 0.0])
CSS2.nHat_B = np.array([0.0, -1.0, 0.0])

CSS1.setBodyToPlatformDCM(90.*macros.D2R, 0., 0.)
CSS1.phi = 90.*macros.D2R
CSS1.theta = 0.*macros.D2R
```

### Solar Eclipse Considerations

An optional input message, `sunEclipseInMsg`, can be used to account for solar eclipses. If not set, the CSS defaults to the spacecraft always being in the sun.

## Modifications Included

This version of the script introduces new command-line options for customizing the simulation, including flags for using CSS constellations, platforms, eclipse considerations, and the Kelly factor. Additionally, parameters for the number of cycles and sensors have been added to provide further flexibility in simulation setups.

---

This `README.md` provides comprehensive instructions for setting up and running the modified Basilisk scenario script, along with detailed information on the parameters and configurations available. Let me know if you need further adjustments!