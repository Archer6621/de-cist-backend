import sys

from retrpred.src.INP_idf_functions.get_retrofit_data2 import get_initial_r_values
from retrpred.src.INP_idf_functions.woon_functions_distr2 import map_archetype
from diskcache import Cache
print("Loading Geomeppy...")
from geomeppy import IDF

from dataloader import load_data
print("Loading Torch...")
from retrpred.src.gen_function_make_geometry_idf_sample_initial import (
    make_geometry_idf_sample_distr,
)
from retrpred.src.niemann import nieman_data
from retrpred.src.surr_function_make_input_vector_and_predict import (
    function_make_input_vector_and_predict,
)
from util import cache_root, get_initialized_idf, get_niemann
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm

print("Done!")

gdf = load_data()

base_prediction_cache = Cache(cache_root / 'prediction')
broken_buildings_cache = Cache(cache_root / 'broken')
pred_existence = set(base_prediction_cache.iterkeys())
niemann_existence = {x for y in nieman_data['BAG Pand ID'].dropna().apply(lambda x : x[1:-1].split(',')) for x in y}
pure_building_cache = Cache(cache_root / 'pure_buildings')

iddfile  = "C:/EnergyPlusV23-1-0/Energy+.idd"
IDF.setiddname(iddfile)


# NOTE: There seems to be some large slowdown when it is calculating the IDF or running the model.
#       It is probably related to the IDF. Given the fact that it currently uses file handles and files
#       in general, doing this in threads is going to be tricky.
#       
#       But the slowdown happens especially when a building errors. It has to load the IDF and do all
#       the underlying wizardry, and the concludes that something is wrong. Will have to ask how to get
#       around it, but for one of the causes I know that it has to do with the niemann data.
# print(niemann_existence)
# counter = -1
# for idx, building in gdf.iterrows():

default_cooling = {"cooling_system":"no_cooling", "dx_cop": 4}
default_heating = {"heating_system": "gas_boiler", "boiler_eff": 0.95, "hp_cop": 5}

def create_init_record(building):
    init_rec = {}
    mapped_archetype = map_archetype(building.n_archetype)
    for component in ["inf_rate","r_wall_ins","r_floor_ins","r_roof_ins","u_window","shgc"]:
        init_rec[component] = get_initial_r_values(mapped_archetype, component)

    init_rec |= default_cooling
    init_rec |= default_heating
    init_rec["sim"] = 0 

    # r_*** == r_***_ins
    init_rec["r_wall"] = init_rec["r_wall_ins"]
    init_rec["r_floor"] = init_rec["r_floor_ins"]
    init_rec["r_roof"] = init_rec["r_roof_ins"]

    # All ageing to 1 
    init_rec["ageing_wall"] = 1
    init_rec["ageing_floor"] = 1
    init_rec["ageing_roof"] = 1

    # All years to 0
    init_rec["year_wall"] = 0
    init_rec["year_floor"] = 0
    init_rec["year_window"] = 0
    init_rec["year_roof"] = 0
    init_rec["year_gas_boiler"] = 0
    init_rec["year_heat_pump"] = 0
    init_rec["year_airco"] = 0

    init_rec["u_glazing_type"] = 2
    return init_rec


def precache(idx, building):
    # counter += 1
    print(f"{0:>10} | ", '[.] Pre-caching:', idx)
    if idx in pred_existence:
        print(f"{0:>10} | ", "[o] Already calculated", idx)
        return
    # TODO: Add option to skip this in commandline args, e.g. --redo-errored
    if idx in broken_buildings_cache:
        print(f"{0:>10} | ", "[s] Errored before, skipping", idx)
        return
    if f'"{idx:0>16}"' not in niemann_existence:
        print(f"{0:>10} | ", "[n] Skipping, no niemann data available:", f'"{idx:0>16}"')
        return
    try:
        (idf, building, records_init_singleb) = get_initialized_idf(int(idx), 1)
        init_rec = [create_init_record(building)]
        first_rec = function_make_input_vector_and_predict(idf, building, init_rec)[0]
        heating = first_rec["heating_dem_m2"]
        cooling = first_rec["cooling_dem_m2"]
        gas = first_rec["heating_gas_m2"]
        elec = first_rec["heating_elec_m2"] + first_rec["cooling_elec_m2"]
        co2 = first_rec["carbon emissions"]
        under = first_rec["avg_underheat_hours"]
        over = first_rec["avg_overheat_hours"]

        base_prediction_cache.set(idx, {"heat": heating, "cool":cooling, "elec": elec,"gas": gas,"co2": co2, "under":under, "over":over})
        
        # Also set the pure building cache.
        pure_building_cache.set(idx, building)

        # Also precache the niemann data.
        _ = get_niemann(idx)

        print(f"{0:>10} | ", "[s] SUCCESS", idx)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit()
    except Exception as e:
        print(f"{0:>10} | ", "[!] Could not process building", idx, type(e).__name__, e)
        broken_buildings_cache.set(idx, f"{type(e).__name__}: {e}");
    
from joblib import Parallel, delayed

with tqdm_joblib(tqdm(desc="Precaching", total=len(gdf.index))) as progress_bar:
    Parallel(n_jobs=16, prefer="threads")(delayed(precache)(idx, building) for idx, building in gdf.iterrows())
