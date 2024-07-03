import pickle
import pandas as pd
import sys

sys.path.append("../Code")
import functions_light_absorption as lip
from parameters import p
from module_electron_transport_chain import complex_absorption
from module_update_phycobilisomes import OCP_absorbed_light

from functions_light_absorption import get_pigment_absorption, light_spectra, get_mean_sample_light
from modelbase.ode import Model
from modelbase.ode import ratefunctions as rf

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
    elif isinstance(rate_ratios, str) and rate_ratios == "sum":
        return res.sum(axis=1)
    elif isinstance(rate_ratios, (pd.Series, dict)):
        rate_ratios = pd.Series(rate_ratios)
        return (res * rate_ratios).sum(axis=1)
    else:
        raise ValueError(f"uncrecognized input for rate_ratios: {rate_ratios}")
    
# Funciton to get the required predictor inputs from Andreas model outputs
def get_model_inputs(
        cell_density, # [cells ml^-1]
        chlorophyll, # [µmol l^-1]
        carotenoids, # [µmol l^-1]
        phycocyanin, # [µmol l^-1] FIXME: Not implemented yet
        allophycocyanin, # [µmol l^-1] FIXME: Not implemented yet
        light_intensity, # Model
        sample_depth_m=0.01, # [m] Assuming a cuvette with 1 cm diameter
        beta_carotene_fraction = 0.26, # [rel] fraction of beta-carotene of cellular carotenoids
        cell_volume = 4e-15, # [l]
    ):
    # Calculate the chlorophyll content
    mg_chlorophyll  = (chlorophyll*893.509/1000) # [mg l^-1]

    # Calculate the relative contents of carotenoids, phycocyanin, and allophycocyanin 
    relative_carotenoids = (carotenoids*581.5565/1000)/mg_chlorophyll # [mg mg(Chla)^-1]  581.5565 is weighted average mol weight of carotenoids
    relative_phycocyanin = 10000/mg_chlorophyll # [mg mg(Chla)^-1] FIXME Arbitrary number for now
    relative_allophycocyanin = 2000/mg_chlorophyll # [mg mg(Chla)^-1] FIXME Arbitrary number for now

    # Get the pigment content of the cell
    pigment_content = pd.Series({
        "chla": 1, # This is always 1
        "beta_carotene":relative_carotenoids,
        "phycocyanin": relative_phycocyanin,
        "allophycocyanin":relative_allophycocyanin
    })

    # Only beta carotene is photosynthetically active
    photopigment_content = pigment_content.copy()
    photopigment_content["beta_carotene"] = photopigment_content["beta_carotene"] * beta_carotene_fraction

    # Get the total cellular absorption
    absorption = get_pigment_absorption(pigment_content)
    absorption = absorption.sum(axis=1)
    
    # Define the input light
    light = light_spectra(which="warm_white_led", intensity=light_intensity) # The model light intensity is µmol(photons) m^-2 s^-1, correct?

    # Calculate the chlorophyll concentration in the sample [mg(Chl) m^-3]
    chlorophyll_sample = (
        chlorophyll # [µmol l^-1] intracellular
        * cell_volume # [l cell^-1]
        * cell_density # [cells ml^-1]
        * 1e6 # [ml m^-3]
        * 1e-3 # [mmol µmol^-1]
        * 893.509 # [g mol^-1]
        )

    # Correct the light for absorption
    corrected_light = get_mean_sample_light(
        I0=light, 
        depth=sample_depth_m, # [m]
        absorption_coef=absorption, # wavelength-specific absorption coefficients for whole cell, summed up from pigments
        chlorophyll_sample = chlorophyll_sample # [mg(Chl) m^-3]
    )
    return {
        "orig_pfd": light,
        "absorption_spectrum": absorption,
        "pfd": corrected_light,
        "pigment_content": photopigment_content
        }

# Wrapper function to get the rate estimations from Andreas model outputs
def get_influx_rate_estimations(
        light_intensity, # Model
        rate_ratios,
        cell_density, # [cells ml^-1]
        chlorophyll, # [µmol l^-1]
        carotenoids, # [µmol l^-1]
        phycocyanin=None, # [µmol l^-1]
        allophycocyanin=None, # [µmol l^-1]
        ps_ratio:float=5.9,
        beta_carotene_fraction = 0.26, # [rel] fraction of beta-carotene of cellular carotenoids
        sample_depth_m=0.01, # [m] Assuming a cuvette with 1 cm diameter
        cell_volume = 4e-15, # [l]
        models=models, # The predictor models
        output_rates=["ATP", "NADPH", "3PGA", "Fd_red"]
    ):
    # Get the inputs into the predictor function
    pred_input = get_model_inputs(
        cell_density=cell_density, # [cells ml^-1]
        chlorophyll=chlorophyll, # [µmol l^-1]
        carotenoids=carotenoids, # [µmol l^-1]
        phycocyanin=phycocyanin, # [µmol l^-1]
        allophycocyanin=allophycocyanin, # [µmol l^-1]
        light_intensity=light_intensity, # Model
        sample_depth_m=sample_depth_m, # [m] Assuming a cuvette with 1 cm diameter
        beta_carotene_fraction=beta_carotene_fraction, # [rel] fraction of beta-carotene of cellular carotenoids
        cell_volume=cell_volume, # [l]
    )

    # Predict the simulated rates
    rates = get_simulated_ssrates(
        light=pred_input["pfd"], 
        pigment_content=pred_input["pigment_content"],
        rate_ratios = rate_ratios,
        ps_ratio=ps_ratio, # Value in the default model
        models=models,
        output_rates=output_rates,
    )

    return rates

# Function to make the predictor usable with modelbase
def get_influx_rate_estimations_modelbase(
        light_intensity, # Model
        ATP_prodratio, # Ratio of maximal ATP produciton that should be outputted
        NADPH_prodratio, # Ratio of maximal NADPH produciton that should be outputted
        _3PGA_prodratio, # Ratio of maximal 3PGA produciton that should be outputted
        Fd_red_prodratio, # Ratio of maximal Fd_red produciton that should be outputted
        cell_density, # [cells ml^-1]
        chlorophyll, # [µmol l^-1]
        carotenoids, # [µmol l^-1]
        phycocyanin=None, # [µmol l^-1]
        allophycocyanin=None, # [µmol l^-1]
        ps_ratio:float=5.9,
        beta_carotene_fraction=0.26, # [rel] fraction of beta-carotene of cellular carotenoids
        sample_depth_m=0.01, # [m] Assuming a cuvette with 1 cm diameter
        cell_volume = 4e-15, # [l]
):
    # Iterate through the given concentration fractions and determine 
    rates = []
    for ATP, NADPH, _3PGA, Fd_red in zip(ATP_prodratio, NADPH_prodratio, _3PGA_prodratio, Fd_red_prodratio):
        _rates = get_influx_rate_estimations(
            light_intensity=light_intensity, # Model
            rate_ratios=pd.Series([ATP, NADPH, _3PGA, Fd_red], index=["ATP", "NADPH", "3PGA", "Fd_red"]),
            cell_density=cell_density, # [cells ml^-1]
            chlorophyll=chlorophyll, # [µmol l^-1]
            carotenoids=carotenoids, # [µmol l^-1]
            phycocyanin=phycocyanin, # [µmol l^-1]
            allophycocyanin=allophycocyanin, # [µmol l^-1]
            beta_carotene_fraction=beta_carotene_fraction, # [rel] fraction of beta-carotene of cellular carotenoids
            ps_ratio=ps_ratio,
            sample_depth_m=sample_depth_m, # [m] Assuming a cuvette with 1 cm diameter
            cell_volume=cell_volume
        )
        rates.append(_rates)

    rates = pd.concat(rates, axis=1)
    return rates.to_numpy()