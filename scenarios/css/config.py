# Monte Carlo configs
NUMBER_OF_RUNS = 100  # per simulation

# Default Configuration for the CSS Scenario
DEFAULT_CSS_CONFIG = {
    # TODO : Define the nHat_B for the all sensors
    "params": [
        {
            "fov": 80.0,
            "r_B": [2.00131, 2.36638, 1.0],
            "scaleFactor": 2.0,
            "phi": 0.0,
            "theta": 45.0,
            "nHat_B": [1.0, 0.0, 0.0],
            "kellyFactor": 0.0,
        },
        {
            "fov": 80.0,
            "r_B": [-3.05, 0.55, 1.0],
            "scaleFactor": 2.0,
            "phi": 90.0,
            "theta": 45.0,
            "nHat_B": [0.0, 1.0, 0.0],
            "kellyFactor": 0.0,
        },
        {
            "fov": 45.0,
            "r_B": [-3.05, 0.55, 1.0],
            "scaleFactor": 2.0,
            "phi": 180.0,
            "theta": 45.0,
            "nHat_B": [-1.0, 0.0, 0.0],
            "kellyFactor": 0.0,
        },
        {
            "fov": 80.0,
            "r_B": [-3.05, 0.55, 1.0],
            "scaleFactor": 2.0,
            "phi": 270.0,
            "theta": 45.0,
            "nHat_B": [1.0, 0.0, 0.0],
            "kellyFactor": 0.0,
        },
        {
            "fov": 80.0,
            "r_B": [-3.05, 0.55, 1.0],
            "scaleFactor": 2.0,
            "phi": 0.0,
            "theta": -45.0,
            "nHat_B": [1.0, 0.0, 0.0],
            "kellyFactor": 0.0,
        },
        {
            "fov": 80.0,
            "r_B": [-3.05, 0.55, 1.0],
            "scaleFactor": 2.0,
            "phi": 90.0,
            "theta": -45.0,
            "nHat_B": [1.0, 0.0, 0.0],
            "kellyFactor": 0.0,
        },
        {
            "fov": 80.0,
            "r_B": [-3.05, 0.55, 1.0],
            "scaleFactor": 2.0,
            "phi": 180.0,
            "theta": -45.0,
            "nHat_B": [1.0, 0.0, 0.0],
            "kellyFactor": 0.0,
        },
        {
            "fov": 80.0,
            "r_B": [-3.05, 0.55, 1.0],
            "scaleFactor": 2.0,
            "phi": 270.0,
            "theta": -45.0,
            "nHat_B": [1.0, 0.0, 0.0],
            "kellyFactor": 0.0,
        },
    ],
    "description": "Default Sensor Configuration",
}

# Tampered Ranges of Input CSS Configurations for Monte Carlo Simulation
TAMPERED_RANGES = [
    {
        "range_max": {
            "fov": 80.0,
            "kellyFactor": 1.0,
            "scaleFactor": 10,
        },
        "range_min": {
            "fov": 0.0,
            "kellyFactor": 0.0,
            "scaleFactor": 0.0,
        },
        "description": "Tampered Sensor Configuration ranges",
    },
]

SIMULATIONS_PARAMETERS = [
    {
        "use_css_constellation": False,
        "use_kelly": False,
        "use_eclipse": False,
        "number_of_cycles": 5,
        "is_tampered_fov": True,
        "is_tampered_scale_factor": False,
        "is_tampered_kelly_factor": False,
        "name" : "Tampered FOV"
    },
   {
        "use_css_constellation": False,
        "use_kelly": False,
        "use_eclipse": False,
        "number_of_cycles": 5,

        "is_tampered_fov": False,
        "is_tampered_scale_factor": True,
        "is_tampered_kelly_factor": False,
        "name" : "Tampered scale factor"
    },
    {
        "use_css_constellation": False,
        "use_kelly": False,
        "use_eclipse": False,
        "number_of_cycles": 5,

        "is_tampered_fov": True,
        "is_tampered_scale_factor": True,
        "is_tampered_kelly_factor": False,
        "name" : "Tampered scale factor and FOV"
    },
    {
        "use_css_constellation": False,
        "use_kelly": False,
        "use_eclipse": True,
        "number_of_cycles": 5,

        "is_tampered_fov": True,
        "is_tampered_scale_factor": False,
        "is_tampered_kelly_factor": False,
        "name" : "Tampered FOV in eclipse"
    },
    {
        "use_css_constellation": False,
        "use_kelly": False,
        "use_eclipse": True,
        "number_of_cycles": 5,

        "is_tampered_fov": False,
        "is_tampered_scale_factor": True,
        "is_tampered_kelly_factor": False,
        "name" : "Tampered scale factor in eclipse"
    },
    {
        "use_css_constellation": False,
        "use_kelly": False,
        "use_eclipse": True,
        "number_of_cycles": 5,

        "is_tampered_fov": True,
        "is_tampered_scale_factor": True,
        "is_tampered_kelly_factor": False,
        "name" : "Tampered scale factor and FOV in eclipse"
    },

]
