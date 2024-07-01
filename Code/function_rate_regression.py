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

from function_concentration_regression import make_light_into_input

# Import the email notifyer
from SMTPMailSender import SMTPMailSender

# Set 
max_workers = 100
max_workers = np.min([max_workers, os.cpu_count() - 2])
file_prefix = f"rateregression_{datetime.now().strftime('%Y%m%d%H%M')}"
n_points = 10

target_compounds = ["ATP", "NADPH", "3PGA", "Fd_red"]

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

# %%
# Generate the input light data
_light_input = np.linspace(10, 1000, n_points)

light_input = np.array(np.meshgrid(
    _light_input,
    _light_input,
    _light_input,
    _light_input,
    range(len(target_compounds)),
)).T.reshape(-1,5)

light_input = pd.DataFrame(
    light_input,
    columns = [
        "complex_abs_ps1",
        "complex_abs_ps2",
        "complex_abs_pbs",
        "light_ocp",
        "target_compound"
    ]
)

light_input["target_compound"] = pd.Series([target_compounds[int(x)] for x in light_input["target_compound"]], dtype=str)

light_input.shape[0]

# Add a sink reaction with Mass Action kinetics to the model
def vsink_hill(k, *X):
    concs = X[:len(X)//2]
    Ks = X[len(X)//2:]

    if len(concs) != len(Ks):
        raise ValueError("differing numbers of concentrations and Hill constants given")

    res = k
    for conc, K in zip(concs, Ks):
        res*= rf.hill(S=conc, vmax=1, kd=K, n=4)

    return res

def vcap(S, k, thresh, lower_cap=False):
    v = np.array(k * (S - thresh))
    v = v.reshape(-1)
    if not lower_cap:
        v[v<0] = 0
    return v

def add_sink_allcap(m, target_compound, all_target_compounds, k=10000):
    # Make a copy of the model, in case the mca adaption version should not be applied to the original model
    m = m.copy()

    # Add caps for all non-target compounds
    for cap in [x for x in all_target_compounds if x != target_compound]:
        # Set the stoichiometry
        if cap == "NADPH":
            cap_stoich = {
            "NADPH": -1,
            "Ho": -1/m.get_parameter("bHo"),
            }
        if cap == "Fd_red":
            cap_stoich = {"Fd_ox": 1}
        else:
            cap_stoich = {cap:-1}

        # Set the cap level
        if cap == "3PGA":
            cap_thresh = 1000
        elif cap == "ATP":
            cap_thresh = 0.95 * m.get_parameter("AP_tot")
        elif cap == "NADPH":
            cap_thresh = 0.95 * m.get_parameter("NADP_tot")
        elif cap == "Fd_red":
            cap_thresh = 0.95 * m.get_parameter("Fd_tot")


        m.add_parameters({
            f"kCap_{cap}": 10000,
            f"threshCap_{cap}": cap_thresh,
            f"lowerCap_{cap}": False,
        })
        
        m.add_reaction_from_args(
            rate_name=f"vout_{cap}",
            function=vcap,
            stoichiometry=cap_stoich,
            args=[cap, f"kCap_{cap}", f"threshCap_{cap}", f"lowerCap_{cap}"]
        )

    # Define the stoichiometry necessary for the sink reaction and the compounds that should be used for the kinetic function
    # Add the sink parameters and cap 3PGA, ATP and NADPH
    m.add_parameters({
        "kSink": k,
    })

    # Set the stoichiometry
    if target_compound == "NADPH":
        stoich = {
        "NADPH": -1,
        "Ho": -1/m.get_parameter("bHo"),
        }
    elif target_compound == "Fd_red":
        stoich = {
        "Fd_ox": 1,
        "Fd_red":-1,
        "Ho": -1/m.get_parameter("bHo"),
        }
    else:
        stoich = {target_compound:-1}
    modelstoich = {k:v for k,v in stoich.items() if k in m.get_compounds()}
    vargs = [k for k,v in stoich.items() if v<0]

    # Define the hill constants for the different compounds
    KHills = {
        # "Q_red":(0.1 * m.parameters["Q_tot"])**4,
        # "PC_red": (0.1 *m.parameters["PC_tot"])**4,
        "Fd_red": (0.1 *m.parameters["Fd_tot"])**4,
        "NADPH": (0.1 *m.parameters["NADP_tot"])**4,
        # "NADH": (0.1 *m.parameters["NAD_tot"])**4,
        "ATP": (0.1 *m.parameters["AP_tot"])**4,
        "3PGA": 1,
        "Ho": (0.001)**4
    }

    if len(modelstoich) != len(vargs):
        raise ValueError(f"stoichiometry unbalanced: {stoich}, {target_compound}")

    # Select ne needed Hill constants
    Kargs = {f"KSink_{comp}":KHills[comp] for comp in vargs}

    # Add the necessary parameters to the model
    m.add_parameters(Kargs)

    m.add_reaction_from_args(
        rate_name=f"vout_{target_compound}",
        function=vsink_hill,
        stoichiometry=modelstoich,
        args=["kSink"] + vargs + list(Kargs.keys())
    )

    return m
    
def get_ss_rates(x, p_keys, all_target_compounds=target_compounds, file_prefix=file_prefix):
    index = x[0]
    p_values = x[1][:-1]
    target_compound = x[1][-1]
    
    # Define the rates that should be measured
    rates = [f"vout_{x}" for x in all_target_compounds]

    # Adapt the model to the target compound
    # Get the default model
    m, y0 = get_model(verbose=False, check_consistency=False)
    m, y0 = make_light_into_input(m, y0)

    # Set the initial 3PGA concentration to zero
    y0["3PGA"] = 0

    # Add sinks and caps to the model
    m = add_sink_allcap(m, target_compound=target_compound, all_target_compounds=all_target_compounds, k=1e5)

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

    # Get the rates 
    if t is not None:
        res = s.get_fluxes_df()
        conc = res.loc[:, rates].iloc[-1]
    else:
        conc = pd.Series(index=rates)

    # Save the residuals
    with open(Path(f"../out/{file_prefix}_intermediates.csv",), "a") as f:
        f.writelines(f"{index},{','.join([str(x) for x in p_values])},{','.join([str(x) for x in conc.to_numpy()])}\n")

    return conc


# Partially populate the function
_get_ss_rates = partial(
    get_ss_rates,
    p_keys=light_input.columns,
)

# # %%
input = light_input.iterrows()# .to_numpy()


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    with ProcessPoolExecutor(max_workers=max_workers) as pe:
        res = list(tqdm(
            pe.map(_get_ss_rates, input),
            total=light_input.shape[0],
            disable=False
        ))

result = pd.concat(res, axis=1).T.reset_index().drop("index", axis=1)
n_successful = np.invert(result.isna().any(axis=1)).sum()

# Save the parameters and results
light_input.to_csv(Path(f"../Results/{file_prefix}_params.csv",))
result.to_csv(Path(f"../Results/{file_prefix}_results.csv",))

email.send_email(
    body=f"Regression run {file_prefix} was successfully finished\n{n_successful} simulations were successful",
    subject="Regression finished"
)