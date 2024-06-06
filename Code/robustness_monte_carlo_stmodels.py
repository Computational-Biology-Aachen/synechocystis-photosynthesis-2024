# %%
#!/usr/bin/python3
import pandas as pd
import numpy as np
import sys
from tqdm.auto import tqdm
import pebble
import pickle
from functools import partial
import warnings
import logging
import traceback

from datetime import datetime
from concurrent import futures
from pathlib import Path

from modelbase.ode import ratefunctions as rf

sys.path.append("../Code")

# Import model functions
from function_residuals import calculate_residuals, setup_logger, residual_relative_weights
from get_current_model import get_model

# Import the email notifyer
from SMTPMailSender import SMTPMailSender

# %%
# Set the number of evaluated samples
n_mutations = 10000
include_default_model = True

# Set the maximum number of parallel threads and the timeout
n_workers = 100 # Maximum number of parallel threads
timeout = 600 # Timeout for each thread in seconds

# Set the prefix to be used for logging and results files
file_prefix = f"montecarlo_{datetime.now().strftime('%Y%m%d%H%M')}"
# file_prefix = f"residuals_test"

# Set the random number generator
rng = np.random.default_rng(2024)

# Setup the email sender
email = SMTPMailSender(
    SMTPserver='mail.gmx.net',
    username='tobiaspfennig@gmx.de',
    default_destination='tobiaspfennig@gmx.de'
)

# %%
parameter_ranges = {
    # "PSIItot": (0.91, 1.1),
    # "PSItot": (0.91, 1.1),
    # "Q_tot": (0.91, 1.1),
    # "PC_tot": (0.91, 1.1),
    # "Fd_tot": (0.91, 1.1),
    # "NADP_tot": (0.91, 1.1),
    # "NAD_tot": (0.91, 1.1),
    # "AP_tot": (0.91, 1.1),
    # "O2ext": (0.91, 1.1),
    # "bHi": (0.91, 1.1),
    # "bHo": (0.91, 1.1),
    # "cf_lumen": (0.91, 1.1),
    # "cf_cytoplasm": (0.91, 1.1),
    "fCin": (0.91, 1.1),
    # "kH0": (0.91, 1.1),
    # "kHst": (0.91, 1.1),
    # "kF": (0.91, 1.1),
    # "k2": (0.91, 1.1),
    # "kPQred": (0.91, 1.1),
    # "kPCox": (0.91, 1.1),
    # "kFdred": (0.91, 1.1),
    "k_F1": (0.91, 1.1),
    # "k_ox1": (0.91, 1.1),
    "k_Q": (0.91, 1.1),
    # "k_NDH": (0.91, 1.1),
    # "k_SDH": (0.91, 1.1),
    # "k_FN_fwd": (0.91, 1.1),
    # "k_FN_rev": (0.91, 1.1),
    "k_pass": (0.91, 1.1),
    "k_aa": (0.91, 1.1),
    # "kRespiration": (0.91, 1.1),
    # "kO2out": (0.91, 1.1),
    # "kCCM": (0.91, 1.1),
    "fluo_influence": (0.91, 1.1),
    # "PBS_free": (0.91, 1.1),
    # "PBS_PS1": (0.91, 1.1),
    # "PBS_PS2": (0.91, 1.1),
    "lcf": (0.91, 1.1),
    "KMPGA": (0.91, 1.1),
    "kATPsynth": (0.91, 1.1),
    # "Pi_mol": (0.91, 1.1),
    # "HPR": (0.91, 1.1),
    "kATPconsumption": (0.91, 1.1),
    "kNADHconsumption": (0.91, 1.1),
    # "vOxy_max": (0.91, 1.1),
    # "KMATP": (0.91, 1.1),
    # "KMNADPH": (0.91, 1.1),
    # "KMCO2": (0.91, 1.1),
    # "KIO2": (0.91, 1.1),
    # "KMO2": (0.91, 1.1),
    # "KICO2": (0.91, 1.1),
    # "vCBB_max": (0.91, 1.1),
    # "kPR": (0.91, 1.1),
    "kUnquench": (0.91, 1.1),
    "KMUnquench": (0.91, 1.1),
    "kQuench": (0.91, 1.1),
    "KHillFdred": (0.91, 1.1),
    "nHillFdred": (0.91, 1.1),
    # "k_O2": (0.91, 1.1),
    # "cChl": (0.91, 1.1),
    # "CO2ext_pp": (0.91, 1.1),
    # "S": (0.91, 1.1),
    "kCBBactivation": (0.91, 1.1),
    "KMFdred": (0.91, 1.1),
    "kOCPactivation": (0.91, 1.1),
    "kOCPdeactivation": (0.91, 1.1),
    "OCPmax": (0.91, 1.1),
    "vNQ_max": (0.91, 1.1),
    "KMNQ_Qox": (0.91, 1.1),
    "KMNQ_Fdred": (0.91, 1.1),
}

# %%
# Load the model to get default parameter values
m = get_model(get_y0=False, verbose=False, check_consistency=False)

# %%
# Define a function to generate a number of random log-spaced factors to be used with the parameters
def get_mutation_factors(n, start, end, rng):
    rand = rng.random(n)
    return np.exp(np.log(start) + rand*(np.log(end)-np.log(start)))

def get_parameter_mutations(n, parameter_ranges, rng, m):
    # Create a container for the mutations
    res = pd.DataFrame(index=np.arange(n), columns=parameter_ranges.keys(), dtype=object)

    for k,v in parameter_ranges.items():
        if k=="fluo_influence":
            _res = {l:m.parameters[k][l] * get_mutation_factors(n, *v, rng=rng) for l in m.parameters[k]}
            res[k] = pd.DataFrame(_res).T.to_dict()
        else:
            # Mutate the default parameter value with the random factors
            res[k] = m.parameters[k] * get_mutation_factors(n, *v, rng=rng)
    return res

# %%
# Define a function to be executed by each thread
def thread_function(x, input_model, input_y0, **kwargs):
    # Unpack the input index and parameter values
    index, p = x

    try:
        # Execute the actual function
        result = calculate_residuals(p, input_model=input_model, input_y0=input_y0, thread_index=index, **kwargs)

        return index, result

        # Handle the result if needed
    except Exception as e:
        warnings.warn(f"An error occurred in thread {index} with parameter {p}: {e}")


# Add other state transition models      
# PBS detachment model
def remove_statetransitions_default(m, y0):
    m.remove_reactions(["vPSIIunquench", "vPSIIquench"])

    # Adapt y0
    y0["PSII"] = m.get_parameter("PSIItot")
    return m, y0

def vPBS_detach(Q_ox, PBS_free, Q_red, kPBS_detach, kPBS_attach, PBS_freemax):
    return rf.reversible_mass_action_2_2((PBS_freemax-PBS_free), Q_ox, PBS_free, Q_red, kPBS_detach, kPBS_attach)

def update_statetransitions_detachpbs1(m, y0):
    # Remove old description
    m, y0 = remove_statetransitions_default(m, y0)

    # Adapt y0
    y0["PBS_PS1"] = 0.45
    y0["PBS_PS2"] = 0.55

    # Add new parameters and functions
    m.add_parameters({
        "kPBS_detach":1e-3,
        "kPBS_attach":1e-3,
        "PBS_freemax":0.5
    })

    m.add_reaction_from_args(
        rate_name="vPBS_detach",
        function=vPBS_detach,
        stoichiometry={"PBS_PS1": -1, "PBS_PS2":-1},
        args=["Q_ox", "PBS_free", "Q_red", "kPBS_detach", "kPBS_attach", "PBS_freemax"],
    )

    return m, y0

# PBS mobile model
def vPBS_mobile(Q_ox, PBS_PS1, Q_red, PBS_PS2, kPBS_toPS1, kPBS_toPS2, PBS_PS1min, PBS_PS2min):
    return rf.reversible_mass_action_2_2(
        (PBS_PS2 - PBS_PS2min), Q_red, 
        (PBS_PS1 - PBS_PS1min), Q_ox, 
        kPBS_toPS1, kPBS_toPS2
    )

def update_statetransitions_mobilepbs(m, y0):
    # Remove old description
    m, y0 = remove_statetransitions_default(m, y0)

    # Adapt y0
    y0["PBS_PS1"] = 0.45
    y0["PBS_PS2"] = 0.55

    # Add new parameters and functions
    m.add_parameters({
        "kPBS_toPS1":1e-3,
        "kPBS_toPS2":1e-3,
        "PBS_PS1min": 0.35, 
        "PBS_PS2min": 0.45,
    })

    m.add_reaction_from_args(
        rate_name="vPBS_mobile",
        function=vPBS_mobile,
        stoichiometry={"PBS_PS1": 1, "PBS_PS2":-1},
        args=["Q_ox", "PBS_PS1", "Q_red", "PBS_PS2", "kPBS_toPS1", "kPBS_toPS2", "PBS_PS1min", "PBS_PS2min"],
    )

    return m, y0

# Spillover model
def vspillover(spill, Q_red, Q_ox, kspill, kunspill, spillmax):
    return rf.reversible_mass_action_2_2(spillmax-spill, Q_red, spill, Q_ox, kspill, kunspill)

def ps_normabsorption_spill(time, PBS_PS1, PBS_PS2, spill, complex_abs, PSItot, PSIItot, lcf):
    light_ps1 = (complex_abs["ps1"] + complex_abs["pbs"] * PBS_PS1) / PSItot
    light_ps2 = (complex_abs["ps2"] + complex_abs["pbs"] * PBS_PS2) / PSIItot

    if isinstance(light_ps2, float) and isinstance(time, np.ndarray):
        light_ps1 = np.repeat(light_ps1, len(time))
        light_ps2 = np.repeat(light_ps2, len(time))

    return (light_ps1 + spill * light_ps2) * lcf, ((1-spill) * light_ps2) * lcf

def update_statetransitions_spillover(m, y0):
    # Remove old description
    m, y0 = remove_statetransitions_default(m, y0)

    # Add new parameters and functions
    m.add_compound("spill")
    y0["spill"] = 0

    m.add_parameters({
        "kspill": 1e-5,
        "kunspill" :1e-5,
        "spillmax": 0.1  
    })

    m.add_reaction_from_args(
        rate_name="vspillover",
        function=vspillover,
        stoichiometry={"spill": 1},
        args=["spill", "Q_red", "Q_ox", "kspill", "kunspill", "spillmax"],
    )

    m.update_algebraic_module(
        module_name="ps_normabsorption",
        function=ps_normabsorption_spill,
        args=["time", "PBS_PS1", "PBS_PS2", "spill", "complex_abs", "PSItot", "PSIItot", "lcf"],
        check_consistency=False,
    )

    m.update_algebraic_module(  # >> changed: added <<
        module_name="ps_normabsorption_ML",
        function=ps_normabsorption_spill,
        args=[
            "time",
            "PBS_PS1",
            "PBS_PS2",
            "spill",
            "complex_abs_ML",
            "PSItot",
            "PSIItot",
            "lcf"
        ],
        check_consistency=False,
    )

    return m, y0

stmodels = {
    "mspill":{
        "pbs_behaviour": "static",
        "fun": update_statetransitions_spillover,
        "param":{
            "kspill": 5e-3,
            "kunspill" :5e-4,
            "spillmax": 0.3 
        },
        "param_bounds":{
            "kspill": (0.91, 1.1),
            "kunspill": (0.91, 1.1),
            "spillmax": (0.91, 1.1) ,
        }
    },
    "mpbsd":{
        "pbs_behaviour": "dynamic",
        "fun": update_statetransitions_detachpbs1,
        "param":{
            "kPBS_detach":1e-4,
            "kPBS_attach":1e-3,
            "PBS_freemax":0.1
        },
        "param_bounds":{
            "kPBS_detach": (0.91, 1.1),
            "kPBS_attach": (0.91, 1.1),
            "PBS_freemax": (0.91, 1.1),
        }
    },
    "mpbsm":{
        "pbs_behaviour": "dynamic",
        "fun": update_statetransitions_mobilepbs,
        "param":{
            "kPBS_toPS1":5e-3,
            "kPBS_toPS2":1e-3,
            "PBS_PS1min": 0.25, 
            "PBS_PS2min": 0.35,
        },
        "param_bounds":{
            "kPBS_toPS1": (0.91, 1.1),
            "kPBS_toPS2": (0.91, 1.1),
            "PBS_PS1min": (0.91, 1.1),
            "PBS_PS2min": (0.91, 1.1)
        }
    }
}

def get_stmodel_and_parameters(chosen_model, stmodels, default_parameters):
    # Create the model with the schosen state transition mechanism
    model = stmodels[chosen_model]

    m,y0 = get_model(check_consistency=False, verbose=False, pbs_behaviour=model["pbs_behaviour"])
    m,y0 = model["fun"](m, y0)
    m.update_parameters(model["param"])

    params = default_parameters.copy()
    params.update(model["param_bounds"])

    return m, y0, params

if __name__ == "__main__":
    # Setup logging
    InfoLogger = InfoLogger = setup_logger("InfoLogger", Path(f"../out/{file_prefix}_info.log"), level=logging.INFO)
    ErrorLogger = setup_logger("ErrorLogger", Path(f"../out/{file_prefix}_err.log"), level=logging.ERROR)


    for stmodel_nam, stmodel in stmodels.items():

        # Make a new file prefix
        _file_prefix = file_prefix+ "_" + stmodel_nam

        # Get the correct model and parameters
        mst, y0st, parameter_rangesst = get_stmodel_and_parameters(stmodel_nam, stmodels, parameter_ranges)

        # Log the start of the run
        InfoLogger.info(f"Started run {stmodel_nam}")

        email.send_email(
            body=f"Monte Carlo run {_file_prefix} was successfully started",
            subject=f"Monte Carlo {stmodel_nam} started"
        )

        # %%
        # Create the parameters
        params = get_parameter_mutations(n_mutations+include_default_model, parameter_rangesst, rng, m=mst)

        # Include the default model in first position
        if include_default_model:
            params.loc[0] = pd.Series({k: mst.parameters[k] for k in params.columns})

        # Save the parameters
        params.to_csv(f"../Results/{_file_prefix}_params.csv")

        # Initialise container for residuals
        results = pd.DataFrame(index=np.arange(n_mutations+include_default_model), columns=(residual_relative_weights.keys()), dtype=float)

        # Catch unnecessary warnings:
        with warnings.catch_warnings() as w:
            # Cause all warnings to always be triggered.
            # warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore")
            
            try:
                # Execute the thread_function for each parameter input
                with tqdm(total=n_mutations+include_default_model) as pbar:
                    with pebble.ProcessPool(max_workers=n_workers) as pool:
                        future = pool.map(
                            partial(
                                thread_function,
                                input_model=mst,
                                input_y0=y0st,
                                intermediate_results_file=f"../out/{_file_prefix}_intermediate.csv",
                                logger_filename=f"../out/{file_prefix}",
                                return_all=True
                            ),
                            params.iterrows(),
                            timeout=timeout,
                        )
                        it = future.result()
                        
                        while True:
                            try:
                                index, res = next(it)
                                pbar.update(1)
                                results.loc[index,:] = res[1]
                            except futures.TimeoutError:
                                pbar.update(1)
                            except StopIteration:
                                break
                            except Exception as e:
                                pbar.update(1)
                                ErrorLogger.error("Error encountered in residuals\n" + str(traceback.format_exc()))
                            finally:
                                pbar.update(1)

                # Save the results
                results.to_csv(f"../Results/{_file_prefix}_results.csv")
                InfoLogger.info(f"Finished run {stmodel_nam} successfully")

                email.send_email(
                    body=f"Monte Carlo run {_file_prefix} finished successfully",
                    subject=f"Monte Carlo {stmodel_nam} successful"
                )
            
            except Exception as e:
                ErrorLogger.error(f"Error encountered in Monte Carlo function on run {stmodel_nam}\n" + str(traceback.format_exc()))
                InfoLogger.info("Finished run with Error")
                email.send_email(
                    body=f"Monte Carlo run {_file_prefix} encountered an Error:\n{e}",
                    subject=f"Monte Carlo {stmodel_nam} Error"
                )