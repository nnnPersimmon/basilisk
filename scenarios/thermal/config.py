# TODO: if we simulate multiple sensors, we should add "type" of module for each sim

# ########################### countermeasures
# ref:`sensorThermal` module to model the temperature of a sensor using radiative heat transfer
DEFAULT_THERMAL_SENSOR_CONFIG = {
    "params": {
        "T_0": 0,  # Celsius
        "nHat_B": [0, 0, 1],
        "sensorArea": 1.0,  # m^2
        "sensorAbsorptivity": 0.25,
        "sensorEmissivity": 0.34,
        "sensorMass": 2.0,  # kg
        "sensorSpecificHeat": 890,
        "sensorPowerDraw": 30.0,  # W
        "pHat_B": [0, 0, -1],
    },
    "description": "Default Sensor Configuration",
}


TAMPERED_IMPUT_SENSOR_CONFIGS = [
    {
        "params": {"T_0": 100},  # Tampered Initial Temperature
        "description": "Tampered Initial Temperature",
    },
    {
        "params": {"nHat_B": [0, 1, 0], "pHat_B": [0, 0, 1]},  # Tampered Orientation
        "description": "Tampered Orientation",
    },
    {
        "params": {"sensorPowerDraw": 100.0},  # Tampered Power Draw
        "description": "Tampered Power Draw",
    },
    {
        "params": {
            "T_0": 100,
            "nHat_B": [0, 1, 0],
            "pHat_B": [0, 0, 1]
        },  # Tampered Initial Temperature and Orientation
        "description": "Tampered Initial Temperature and Orientation",
    },
    {
        "params": {
            "T_0": 100,
            "sensorPowerDraw": 100.0,
        },  # Tampered Initial Temperature and Power Draw
        "description": "Tampered Initial Temperature and Power Draw",
    },
    {
        "params": {
            "nHat_B": [0, 1, 0],
            "pHat_B": [0, 0, 1],
            "sensorPowerDraw": 100.0,
        },  # Tampered Orientation and Power Draw
        "description": "Tampered Orientation and Power Draw",
    },
    {
        "params": {
            "T_0": 100,
            "nHat_B": [0, 1, 0],
            "pHat_B": [0, 0, 1],
            "sensorPowerDraw": 100.0,
        },  # Tampered All Three
        "description": "Tampered All Three",
    },
]


TAMPERED_OUTPUT_SENSOR_CONFIGS = [
    {
        "params": {"pre_switch_delay": 0, "post_switch_delay": 0},
        "description": "No delay before or after orientation switch",
    },
    {
        "params": {"pre_switch_delay": 500, "post_switch_delay": 0},
        "description": "500 ms delay before orientation switch, no delay after",
    },
    {
        "params": {"pre_switch_delay": 0, "post_switch_delay": 500},
        "description": "500 ms delay after orientation switch, no delay before",
    },
    {
        "params": {"pre_switch_delay": 1000, "post_switch_delay": 1000},
        "description": "1000 ms delay before and after orientation switch",
    },
    {
        "params": {"pre_switch_delay": 2000, "post_switch_delay": 1000},
        "description": "2000 ms delay before orientation switch, 1000 ms delay after",
    },
    {
        "params": {"pre_switch_delay": 1000, "post_switch_delay": 2000},
        "description": "1000 ms delay before orientation switch, 2000 ms delay after",
    },
    {
        "params": {"pre_switch_delay": 5000, "post_switch_delay": 5000},
        "description": "5000 ms delay before and after orientation switch",
    },
]
