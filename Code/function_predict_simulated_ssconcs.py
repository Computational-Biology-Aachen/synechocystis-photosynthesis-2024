import pickle
import pandas as pd
import sys

sys.path.append("../Code")
import functions_light_absorption as lip
from parameters import p
from module_electron_transport_chain import complex_absorption
from module_update_phycobilisomes import OCP_absorbed_light

# Load the regression model
with open("../Results/concentration_regression_model.pickle", "rb") as f:
    model = pickle.load(f)

# Define the function that retrieves the predicted concentrations
def get_simulated_ssconcentrations(
        light:pd.Series, 
        ps_ratio:float=5.9, # Value in the default model
        pigment_content=p["pigment_content"],
        model=model,
        input_params=['complex_abs_ps1', 'complex_abs_ps2', 'complex_abs_pbs', 'light_ocp'],
        output_compounds=["ATP", "NADPH", "3PGA", "Fd_red"]
    ):
    # Calculate the absorbed lights from the light input
    complex_abs = complex_absorption(light, ps_ratio, pigment_content) 
    light_params = {f"complex_abs_{k}":v for k,v in complex_abs.items()}

    # Calculate the OCP absorbed light
    light_params["light_ocp"] = OCP_absorbed_light(light)

    # get the input parameters for the model
    model_input = pd.DataFrame(light_params, index=[1])

    # Predict the model simulated ss concentrations
    res = model.predict(model_input)
    res = pd.DataFrame(res, columns=output_compounds).iloc[0]
    res.name = "simulated steady-state concentrations"

    return res