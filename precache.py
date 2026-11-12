import sys

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
from util import cache_root, get_initialized_idf


print("Done!")

gdf = load_data()

base_prediction_cache = Cache(cache_root / 'prediction')
broken_buildings_cache = Cache(cache_root / 'broken')
pred_existence = set(base_prediction_cache.iterkeys())
niemann_existence = {x for y in nieman_data['BAG Pand ID'].dropna().apply(lambda x : x[1:-1].split(',')) for x in y}

iddfile  = "C:/EnergyPlusV23-1-0/Energy+.idd"
IDF.setiddname(iddfile)


# NOTE: There seems to be some large slowdown when it is calculating the IDF or running the model.
#       It is probably related to the IDF. Given the fact that it currently uses file handles and files
#       in general, doing this in threads is going to be tricky.
#       
#       But the slowdown happens especially when a building errors. It has to load the IDF and do all
#       the underlying wizardry, and the concludes that something is wrong. Will have to ask how to get
#       around it, but for one of the causes I know that it has to do with the niemann data.
print(niemann_existence)
counter = -1
for idx, building in gdf.iterrows():
    counter += 1
    print(f"{counter:>10} | ", '[.] Pre-caching:', idx)
    if idx in pred_existence:
        print(f"{counter:>10} | ", "[o] Already calculated", idx)
        continue
    if f'"{idx:0>16}"' not in niemann_existence:
        print(f"{counter:>10} | ", "[n] Skipping, no niemann data available:", f'"{idx:0>16}"')
        continue
    try:
        # TODO: Should probably move this somewhere...
        (idf, building, records_init_singleb) = get_initialized_idf(int(idx), 2)
        first_rec = function_make_input_vector_and_predict(idf, building, records_init_singleb)[0]
        heating = first_rec["heating_dem_m2"]
        cooling = first_rec["cooling_dem_m2"]
        gas = first_rec["heating_gas_m2"]
        elec = first_rec["heating_elec_m2"]
        co2 = first_rec["carbon emissions"]

        base_prediction_cache.set(idx, {"heat": heating, "cool":cooling, "elec": elec,"gas": gas,"co2": co2})
        print(f"{counter:>10} | ", "[s] SUCCESS", idx)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit()
    except Exception as e:
        print(f"{counter:>10} | ", "[!] Could not process building", idx, type(e).__name__, e)
        broken_buildings_cache.set(idx, f"{type(e).__name__}: {e}");
    
