import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import warnings

from modelbase.ode import Simulator
from modelbase.ode import ratefunctions as rf

sys.path.append("../Code")
import functions as fnc
import calculate_parameters_restruct as prm
import functions_light_absorption as lip

# Import model functions
from get_current_model import get_model

from functions_custom_steady_state_simulator import simulate_to_steady_state_custom
from scipy.integrate import simpson
import parameters

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Import the email notifyer
from SMTPMailSender import SMTPMailSender

# Set 
max_workers = 100
max_workers = np.min([max_workers, os.cpu_count() - 2])
file_prefix = f"concregression_{datetime.now().strftime('%Y%m%d%H%M')}"
n_points = 3

# Setup the email sender
email = SMTPMailSender(
    SMTPserver='mail.gmx.net',
    username='tobiaspfennig@gmx.de',
    default_destination='tobiaspfennig@gmx.de'
)

email.send_email(
    body=f"Regression run {file_prefix} was successfully started",
    subject="Regression started"
)

# Define OCP activation as active, light involved process with passive reversal
def OCPactivation(
    OCP, light_ocp, kOCPactivation, kOCPdeactivation, lcf, OCPmax=1
):  # >> changed: added <<
    return (
        light_ocp * lcf * kOCPactivation * (OCPmax - OCP)
        - kOCPdeactivation * OCP
    )


# Add OCP effect on light absorption
def ps_normabsorption_ocp(time, PBS_PS1, PBS_PS2, OCP, complex_abs_ps1, complex_abs_ps2, complex_abs_pbs, PSItot, PSIItot, lcf):
    light_ps1 = (complex_abs_ps1 + complex_abs_pbs * PBS_PS1 * (1 - OCP)) / PSItot
    light_ps2 = (
        complex_abs_ps2 + complex_abs_pbs * PBS_PS2 * (1 - OCP)
    ) / PSIItot

    if isinstance(light_ps2, float) and isinstance(
        time, np.ndarray
    ):  # >> changed: added <<
        light_ps1 = np.repeat(light_ps1, len(time))
        light_ps2 = np.repeat(light_ps2, len(time))

    return light_ps1 * lcf, light_ps2 * lcf  # Get float values


def make_light_into_input(m, y0={}, init_param=None, verbose=True):
    if verbose:
        print("making absorption direct input")

    # Add the light parameters that were added as direct inputs
    p = {
        "complex_abs_ps1": 1,
        "complex_abs_ps2": 1,
        "complex_abs_pbs": 1,
        "light_ocp": 1,
    }

    m.add_parameters(p)

    # Remove unnecessary reactions and parameters
    m.remove_derived_parameter("complex_abs")
    m.remove_parameter("pfd")
    

    # Add OCP activation
    m.update_reaction_from_args(  # >> changed: added <<
        rate_name="OCPactivation",
        function=OCPactivation,
        stoichiometry={"OCP": 1},
        args=["OCP", "light_ocp", "kOCPactivation", "kOCPdeactivation", "lcf", "OCPmax"],
    )

    # >> changed: replaced calculate_excite_ps and the depricated light function with an updated ps_normabsorption <<
    # Add the calculation of normalised absorption by the photosystems
    # Includes PBS association and OCP
    m.update_algebraic_module(
        module_name="ps_normabsorption",
        function=ps_normabsorption_ocp,
        args=["time", "PBS_PS1", "PBS_PS2", "OCP", "complex_abs_ps1", "complex_abs_ps2", "complex_abs_pbs", "PSItot", "PSIItot", "lcf"],
        check_consistency=False,
    )

    # Sort the algebraic modules (temporary until bug is fixed)
    # m = sort_algmodules(m)
    return m, y0

# Get the model and add the biomass reaction
# Get the default model
_m = get_model(get_y0=False, verbose=False, check_consistency=False)

stoich_3PGA = pd.Series({
    "CO2": -3,
    "ATP": -8,
    "NADPH": -5,
    "3PGA": 1,
})

# Biomass stoichiometry as given in Knoop2015
# This reaction has to complete once for a full division
biomass_Knoop2015 = pd.Series({ # [mmol gDW(Biomass)^-1]
    "NADPH": -90, 
    "ATP": -191,
    "CO2": -41.5,
    
    "3PGA": 0, # Added for compliance with the model
})

# Transform to usage of 3PGA
biomass_Knoop2015 = biomass_Knoop2015 - (stoich_3PGA / stoich_3PGA["CO2"] * biomass_Knoop2015["CO2"])
biomass_Knoop2015 = biomass_Knoop2015.drop("CO2")
biomass_Knoop2015["Ho"] = biomass_Knoop2015["NADPH"] / _m.get_parameter("bHo")

# Calculate the conversion from gDW to mol(Chl)
mChl_DW = 10 # [mg gDW^-1] (Zavrel2019)
MChl = 893.509 # [g mol^-1]

nChl_DW = ( # [mol gDW^-1]
    mChl_DW 
    * 1e-3 # [g mg^-1]
    / MChl # [g mol^-1]
)

# Scale the biomass stoichiometry to mol Chl
biomass_Knoop2015 = biomass_Knoop2015 / nChl_DW # [mmol mol(Chl)^-1]


# Get the growth rate of the 633nm (red)-light grown cells
growth_rate= pd.read_csv("../Code/data/Strainparameters_Zavrel2021.csv", header=0,skiprows=1, index_col=0).loc["Specific growth rate","633"] # [h^-1]
growth_rate = growth_rate / 3600 # [s^-1]
# Get the default model
m, y0 = get_model(verbose=False, check_consistency=False)

# Set the initial 3PGA concentration to zero
y0["3PGA"] = 0

# Add a biomass reaction as Mass action
biomass_stoich = biomass_Knoop2015.to_dict()

# Add parameter(s) and reaction to model
m.add_parameter("k_biomass", growth_rate * 10)
m.add_reaction_from_args(
    "vBiomass",
    function = rf.mass_action_variable,
    stoichiometry = biomass_stoich,
    args = ["k_biomass"] + list(biomass_stoich.keys()),
)

m, y0 = make_light_into_input(m, y0)

# %%
# Generate the input light data
_light_input = np.linspace(10, 1000, n_points)

light_input = np.array(np.meshgrid(
    _light_input,
    _light_input,
    _light_input,
    _light_input,
)).T.reshape(-1,4)

light_input = pd.DataFrame(
    light_input, 
    columns = [
        "complex_abs_ps1", 
        "complex_abs_ps2", 
        "complex_abs_pbs", 
        "light_ocp"
    ]
)

light_input.shape[0]

# %%
def get_outputs(x, p_keys, m, y0, compounds=["ATP", "NADPH", "3PGA", "Fd_red"], file_prefix=file_prefix):
    index, p_values = x

    # Adapt and initialise the simulator
    s = Simulator(m)
    p = dict(zip(p_keys, p_values.to_numpy()))
    s.update_parameters(p)
    s.initialise(y0)

    # print(index, p)

    integrator_kwargs = {
        "maxsteps": 20000,
        "atol": 1e-9,
        "rtol": 1e-9,
        "maxnef": 10,
        "maxncf": 10,
    }

    # Simulate to steady state
    s, t, y = simulate_to_steady_state_custom(
        s,
        simulation_kwargs={
            "t_end": 1e6,
            "tolerances": [[["CBBa", "PSII", "OCP"], 1e-8], [None, 1e-6]],
            "verbose": False,
        },
        rel_norm=True,
        return_simulator=True,
        **integrator_kwargs,
    )

    # Get the concentrations 
    if t is not None:
        res = s.get_full_results_df()
        conc = res.loc[:, compounds].iloc[-1]
    else:
        conc = pd.Series(index=compounds)

    # Save the residuals
    with open(Path(f"../out/{file_prefix}_intermediates.csv",), "a") as f:
        f.writelines(f"{index},{','.join([str(x) for x in p_values])},{','.join([str(x) for x in conc.to_numpy()])}\n")

    return conc

# Partially populate the function
_get_outputs = partial(
    get_outputs,
    p_keys=light_input.columns,
    m=m,
    y0=y0,
)

# %%
input = light_input.iterrows()# .to_numpy()


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    with ProcessPoolExecutor(max_workers=max_workers) as pe:
        res = list(tqdm(
            pe.map(_get_outputs, input),
            total=light_input.shape[0],
            disable=False
        ))

result = pd.concat(res, axis=1).T.reset_index().drop("index", axis=1)
n_successful = n_points-result.isna().any(axis=1).sum()

# Save the parameters and results
light_input.to_csv(Path(f"../Results/{file_prefix}_params.csv",))
result.to_csv(Path(f"../Results/{file_prefix}_results.csv",))

email.send_email(
    body=f"Regression run {file_prefix} was successfully finished\n{n_successful} simulations were successful",
    subject="Regression finished"
)