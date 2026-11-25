from typing import Literal, Optional
from fastapi import FastAPI, Response
import geopandas as gpd
import json
# from pyogrio import set_gdal_config_options
from precache import create_init_record, default_cooling, default_heating
from pydantic import BaseModel
from shapely import Polygon, convex_hull
from pathlib import Path
from diskcache import Cache

from retrpred.src.surr_function_make_input_vector_and_predict import function_make_input_vector_and_predict
from retrpred.src.surr_function_make_input_vector_and_predict_over_time import function_make_input_vector_and_predict_over_time
from retrpred.src.gen_function_make_geometry_idf_sample_initial import make_geometry_idf_sample_distr
from retrpred.src.niemann import nieman_data
from retrpred.src.INP_idf_functions.getdata_functions_building import getNieman_data
from retrpred.src.INP_idf_functions.woon_functions_distr2 import map_archetype
from geomeppy import IDF

import matplotlib.pyplot as plt
import io
import numpy as np

from dataloader import load_data
from contextlib import asynccontextmanager

from util import BigCache, cache_root, get_initialized_idf, get_niemann

class ActiveRetrofit(BaseModel):
    name: str
    retro_type: Literal['heating', 'cooling']
    params: None | dict[str, float] = None


class PassiveRetrofit(BaseModel):
    name: str | None
    values: dict[str, float]


class BuildingsRequest(BaseModel):
    coordinates: list[tuple[float, float]]
    square_id: None | int = None

class PredictRequest(BaseModel):
    buildings: list[int]
    r_package: PassiveRetrofit
    r_active: list[ActiveRetrofit]
    square_id: None | int = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gdf = load_data()
    yield

app = FastAPI(lifespan=lifespan)



caches: dict[str, Cache] = dict()

base_prediction_cache = BigCache(cache_root / 'prediction')
broken_buildings_cache = BigCache(cache_root / 'broken')
pure_building_cache = BigCache(cache_root / 'pure_buildings')
pred_existence = set(base_prediction_cache.iterkeys())
broken = set(broken_buildings_cache.iterkeys())

niemann_existence = {int(x.replace('"','')) for y in nieman_data['BAG Pand ID'].dropna().apply(lambda x : x[1:-1].split(',')) for x in y}


iddfile  = "C:/EnergyPlusV23-1-0/Energy+.idd"
IDF.setiddname(iddfile)

n_sim=1




# NOTE: Discovered several fairly large problems that prevent this from working:
#       - IDF has several class attributes that get set, that act as caches. By loading the IDF
#         from a pickle this is circumvented, and thus the whole ordeal crashes when looking for
#         these class attributes. Of course, it is kinda bad that there is no fallback for this that
#         initializes these caches on the fly, but oh well.
#         
#       - Secondly, the whole IDF thing works with files... That get saved... That's most definitely
#         not going to be thread safe. It seems that geomeppy supports file handles though, so we can use
#         BytesIO objects, and those can possibly also be used for memoization.
#         
#         
#         Maybe a quick workaround for now is just to copy memoized IDFs?


@app.get("/api/hello")
def hello( user_id: str ):
    _ = user_cache(user_id)
    return {"Hello": user_id}

def user_cache(user_id: str):
    if user_id not in caches:
        caches[user_id] = Cache(cache_root / user_id)
        # # Initialize some values
        # caches[user_id][""]
    return caches[user_id]

# @router.post("/items/")
# def create_item(item: Item):
#     return {"message": "Item created"}


@app.get("/api/retrofits/")
def get_retrofitted_buildings(user_id: str):
    cache = user_cache(user_id)
    if cache:
        # indices = list(f"{x:0>16}" for x in cache.iterkeys())
        # buildings = app.state.gdf.loc[indices]
        return json.dumps({k:cache[k] for k in cache if "retrofit" in cache[k]})
    return "{}"  # TODO: Check whether this is sensible.


@app.post("/api/buildings/")
def get_buildings(req: BuildingsRequest,  user_id: str):
    # clipped = gdf_wgs84.clip(Polygon(req.coordinates))
    broken = set(broken_buildings_cache.iterkeys())
    indices: set[int] = {x for x in app.state.gdf.index[app.state.gdf.sindex.query(convex_hull(Polygon(req.coordinates)), predicate="intersects")]}
    # print('??', indices)
    indices &= niemann_existence
    indices -= broken
    print("Getting buildings...", indices)
    # Perform checks against Niemann data and exclude.
    exclude = set()
    
    for idx in indices:
        # bag_pand_id = f"{idx:0>16}"
        # building_data = nieman_data[nieman_data['BAG Pand ID'] == bag_pand_id] # querying the data for the specific building unit
        # bag_adres_id = building_data['BAG Adres ID'].values[0] if not building_data.empty else None     
        building_data = get_niemann(idx)
        if building_data is not None:
            print(f"checking {idx}", building_data['Nieman Typologie'], np.array([9, 10, 11, 12]), building_data['Nieman Typologie'] in np.array([9, 10, 11, 12]) )
            if not np.isnan(building_data['Nieman Typologie']) and int(building_data['Nieman Typologie']) in [9, 10, 11, 12]:
                exclude.add(idx)
            if "woonfunctie" not in building_data['Gebruiksdoel']:
                exclude.add(idx)

    intersected: gpd.GeoDataFrame = app.state.gdf.loc[list(indices-exclude)]


    # print(len(indices), len(broken), len(indices-broken))
    # Add predicted features:
    ucache = user_cache(user_id)

    for idx in indices:

        print(idx, type(idx))
        # print(set(user_cache(user_id).iterkeys()))
        init: dict[str, float] = base_prediction_cache.get(idx, {})
        user: dict[str, float] = ucache.get(int(idx), {})
        if int(idx) not in pure_building_cache:
            try:
                pure_building_cache[int(idx)] = get_initialized_idf(int(idx), n_sim)[1]
                
            except Exception as e:
                print(f"Could not retrieve idf/building for '{idx}', marking as broken.")
                broken_buildings_cache[int(idx)] = f"{type(e).__name__}: {e}"
                broken.add(idx)
                continue
        building = pure_building_cache[idx]
        print("Loaded building", idx)

        # Add building-specific data
        intersected.loc[idx, "total_floor_area"] = building.n_total_floor_area
        intersected.loc[idx, "ground_floor_area"] = building.bag_ground_floor_area
        intersected.loc[idx, "pp_ht_floor"] = building.pp_ht_floor
        intersected.loc[idx, "compact_ratio"] = building.pp_compactness_ratio
        intersected.loc[idx, "adiab_wall_area"] = building.bag_area_walls_adiab
        intersected.loc[idx, "archetype"] = map_archetype(building.n_archetype)

        for attr in ['heat', 'cool', 'elec', 'gas', 'co2', 'under', 'over']:
            intersected.loc[idx, f"init_{attr}"] = init.get(attr, 0)
            intersected.loc[idx, f"pred_{attr}"] = user.get(attr, 0)

        if "retrofit" in user:
            intersected.loc[idx, "has_retrofit"] = True
            intersected.loc[idx, "retrofit_details"] = json.dumps(user["retrofit"])

        # NOTE: For some buildings this will change around, because they
        #       are on square boundaries. This is a non-issue for now.
        if req.square_id:
            intersected.loc[idx, "square_id"] = req.square_id



    return intersected.loc[list(indices-exclude-broken)].to_json()


@app.get("/api/reset/")
def reset_user(user_id: str):
    user_cache(user_id).clear()

@app.post("/api/predict/")
def get_predictions(req: PredictRequest,  user_id: str):
    res = {}
    print("setting up")
    # Default to 1 year of sim
    building_lifetime = 2

    # Put together retrofit plan
    components = ["wall", "roof", "floor", "window", "heating_system", "cooling_system", "inf_rate"]
    retrofit_plan: dict[str, list[str|float]] = {component: ["do_nothing"] * 2 for component in components}
    for k in req.r_package.values:
        retrofit_plan[k][1] = f"retrofit_{req.r_package.values[k]}"

        # NOTE: Temporary exception, as it otherwise breaks..
        if k == "window":
            retrofit_plan[k][1] = f"retrofit_{int(req.r_package.values[k])}"

    # NOTE: Not necessary. The defaults are already obtained from "create_init_record",
    #       and the frontend only sends deviations from the default. The only important thing
    #       is that these defaults get synced between the two somehow.
    # 
    # retrofit_plan["heating_system"][1] = f"retrofit_{default_heating['heating_system']}"
    # retrofit_plan["cooling_system"][1] = f"retrofit_{default_cooling['cooling_system']}"

    for retrofit in req.r_active:
        system = f"{retrofit.retro_type}_system"
        retrofit_plan[system][1] = retrofit.name
        print("setting", system, "to", retrofit.name)
        # NOTE: Not sure if/where this actually needs to be set, but we have it just in case.
        # if retrofit.params:
        #     for k in retrofit.params:
        #         retrofit_plan[k][1] = retrofit.params[k]



    cache = user_cache(user_id)
    for bag_id in req.buildings:
        # print("INPUT:", idf, building, records_init_singleb, building_lifetime, retrofit_plan)
        if req.r_package.name !=  None or req.r_active:
            print("getting IDF")
            (idf, building, records_init_singleb) = get_initialized_idf(bag_id, n_sim)
            
            # print("creating initial recrd")
            init_rec  = [create_init_record(building)]

            print("INITIAL STATE:")
            print(init_rec)
            print("RETROFIT TO APPLY:")
            print(retrofit_plan)

            print("predicting..")
            rec = function_make_input_vector_and_predict_over_time(idf, building, init_rec, building_lifetime, retrofit_plan)[-1]
            # print("OUTPUT:", recs)
            print("pred finished!")
            # recs = function_make_input_vector_and_predict(idf, building, records_init_singleb)[0]  # For now, just take the first.
            heating = rec['heating_dem_m2']
            cooling = rec['cooling_dem_m2']
            gas = rec["heating_gas_m2"]
            elec = rec["heating_elec_m2"] + rec["cooling_elec_m2"]
            co2 = rec["carbon_emissions"]
            under = rec["avg_underheat_hours"]
            over = rec["avg_overheat_hours"]

            # Unify this somehow... We either use zero-prefixed strings, or we use integers
            # fixed_bag_id = f"{bag_id:0>16}"
            original: dict[str, float] | None = base_prediction_cache.get(bag_id, None)

            # if original and "cooling_demand" in original and "heating_demand" in original:
            #     res[bag_id] |= {"delta": {"heating_demand": heating-original["heating_demand"], "cooling_demand": cooling-original["cooling_demand"]}}

            preds: dict = {"heat": heating, "cool":cooling, "elec": elec,"gas": gas,"co2": co2, "under": under, "over": over}
            print("predicted", preds)
        else:
            preds: dict = {}
        
        retr = {}
        meta = {}

        if req.r_package.name !=  None or req.r_active:
            retr = {"retrofit" : {"passive": req.r_package.model_dump(), "active": [x.model_dump() for x in req.r_active]}}
            if req.square_id:
                meta["square_id"] = req.square_id
        cache[bag_id] = preds | retr | meta
        res[bag_id] = preds | retr | meta

    return json.dumps(res)


@app.get("/api/idf/")
def get_idf_image(user_id: str, bag_id: int):
    (idf, building, records_init_singleb) = get_initialized_idf(bag_id, n_sim)
    idf.view_model(test=True)

    f = io.BytesIO()
    plt.xlabel("(m)")
    plt.ylabel("(m)")
    plt.gcf().savefig(f, format = 'png')
    f.seek(0)
    return Response(content=f.read(), media_type="image/png")
