import pickle
import pandas as pd
import sys

sys.path.append("../Code")
import functions_light_absorption as lip
from parameters import p
from module_electron_transport_chain import complex_absorption
from module_update_phycobilisomes import OCP_absorbed_light

# Load the regression model
with open("../Results/rate_regression_model.pickle", "rb") as f:
    models = pickle.load(f)

import pickle
import pandas as pd
import sys

sys.path.append("../Code")
import functions_light_absorption as lip
from parameters import p
from module_electron_transport_chain import complex_absorption
from module_update_phycobilisomes import OCP_absorbed_light

# Load the regression model
with open("../Results/rate_regression_model.pickle", "rb") as f:
    models = pickle.load(f)

# Define the function that retrieves the predicted concentrations
def get_simulated_ssrates(
    light:pd.Series, 
    pigment_content=p["pigment_content"],
    rate_ratios = None,
    ps_ratio:float=5.9, # Value in the default model
    models=models,
    output_rates=["ATP", "NADPH", "3PGA", "Fd_red"],
):
    # Calculate the absorbed lights from the light input
    complex_abs = complex_absorption(light, ps_ratio, pigment_content)
    light_params = {f"complex_abs_{k}":v for k,v in complex_abs.items()}

    # Calculate the OCP absorbed light
    light_params["light_ocp"] = OCP_absorbed_light(light)

    # get the input parameters for the model
    model_input = light_params.copy()

    # Include the pigment content
    pigment_input = pigment_content.loc[["phycocyanin", "allophycocyanin", "beta_carotene"]].to_dict()
    pigment_input = {f"pigment_{k}":v for k,v in pigment_input.items()}
    model_input.update(pigment_input)

    # return model_input
    model_input = pd.DataFrame(model_input, index=[1])

    # Create a container for the results
    res = pd.DataFrame(
        columns=pd.Index(models.keys(), name="target_compound"),
        index=pd.Index(output_rates, name="production_rate")
    )

    for nam, model in models.items():
        # Predict the model simulated ss rates
        res[nam] = model.predict(model_input).flatten()

    # Return the ratios
    if rate_ratios is None:
        return res
    elif rate_ratios == "sum":
        return res.sum(axis=1)
    elif isinstance(rate_ratios, (pd.Series, dict)):
        rate_ratios = pd.Series(rate_ratios)
        return (res * rate_ratios).sum(axis=1)