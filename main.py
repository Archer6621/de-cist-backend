from typing import Literal, Optional
from fastapi import FastAPI, Response
import geopandas as gpd
import json
# from pyogrio import set_gdal_config_options
from pydantic import BaseModel
from shapely import Polygon, convex_hull
from pathlib import Path
from diskcache import Cache

from retrpred.src.surr_function_make_input_vector_and_predict import function_make_input_vector_and_predict
from retrpred.src.surr_function_make_input_vector_and_predict_over_time import function_make_input_vector_and_predict_over_time
from retrpred.src.gen_function_make_geometry_idf_sample_initial import make_geometry_idf_sample_distr
from retrpred.src.niemann import nieman_data
from retrpred.src.ep_function_make_enriched_idf import make_enriched_idf

from geomeppy import IDF

import matplotlib.pyplot as plt
import io

from dataloader import load_data
from contextlib import asynccontextmanager

from util import cache_root, get_initialized_idf

class ActiveRetrofit(BaseModel):
    name: str
    retro_type: Literal['heating', 'cooling']
    params: None | dict[str, float] = None

class BuildingsRequest(BaseModel):
    coordinates: list[tuple[float, float]]

class PredictRequest(BaseModel):
    buildings: list[int]
    r_package: dict[str, float]
    r_active: list[ActiveRetrofit]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gdf = load_data()
    yield

app = FastAPI(lifespan=lifespan)



caches: dict[str, Cache] = dict()

base_prediction_cache = Cache(cache_root / 'prediction')
broken_buildings_cache = Cache(cache_root / 'broken')
pred_existence = set(base_prediction_cache.iterkeys())
broken = set(broken_buildings_cache.iterkeys())

niemann_existence = {x.replace('"','') for y in nieman_data['BAG Pand ID'].dropna().apply(lambda x : x[1:-1].split(',')) for x in y}


iddfile  = "C:/EnergyPlusV23-1-0/Energy+.idd"
IDF.setiddname(iddfile)





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
    return caches[user_id]

# @router.post("/items/")
# def create_item(item: Item):
#     return {"message": "Item created"}

@app.post("/api/buildings/")
def get_buildings(req: BuildingsRequest,  user_id: str):
    # clipped = gdf_wgs84.clip(Polygon(req.coordinates))
    indices = {x for x in app.state.gdf.index[app.state.gdf.sindex.query(convex_hull(Polygon(req.coordinates)), predicate="intersects")]}
    indices &= niemann_existence
    indices -= broken
    intersected: gpd.GeoDataFrame = app.state.gdf.loc[list(indices)]

    print(len(indices), len(broken), len(indices-broken))
    # Add predicted features:
    for idx in indices:
        # print(idx)
        # print(set(user_cache(user_id).iterkeys()))
        init: dict[str, float] = base_prediction_cache.get(idx, {})
        user: dict[str, float] = user_cache(user_id).get(int(idx), {})
        for attr in ['heat', 'cool', 'elec', 'gas', 'co2']:
            intersected.loc[idx, f"init_{attr}"] = init.get(attr, 0)
            intersected.loc[idx, f"pred_{attr}"] = user.get(attr, init.get(attr, 0))

        if "retrofit" in user:
            intersected.loc[idx, "has_retrofit"] = True
            intersected.loc[idx, "retrofit_details"] = json.dumps(user["retrofit"])


    return intersected.to_json()


@app.post("/api/predict/")
def get_predictions(req: PredictRequest,  user_id: str):
    res = {}

    # Default to 1 year
    building_lifetime = 2

    # Put together retrofit plan
    components = ["wall", "roof", "floor", "window", "heating_system", "cooling_system", "inf_rate"]
    retrofit_plan: dict[str, list[str|float]] = {component: ["do_nothing"] * 70 for component in components}
    for k in req.r_package:
        retrofit_plan[k][1] = f"retrofit_{req.r_package[k]}"

        # NOTE: Temporary exception...
        if k == "window":
            retrofit_plan[k][1] = f"retrofit_{int(req.r_package[k])}"

    for retrofit in req.r_active:
        system = f"{retrofit.retro_type}_system"
        retrofit_plan[system][1] = retrofit.name
        print("setting", system, "to", retrofit.name)
        # NOTE: Not sure if/where this actually needs to be set, but we have it just in case.
        # if retrofit.params:
        #     for k in retrofit.params:
        #         retrofit_plan[k][1] = retrofit.params[k]



    retrofit_plan["heating_system"][1] = "retrofit_heat_pump"
    retrofit_plan["cooling_system"][1] = "retrofit_dx_airco"


    for bag_id in req.buildings:
        n_sim=2
        (idf, building, records_init_singleb) = get_initialized_idf(bag_id, n_sim)

        # print("INPUT:", idf, building, records_init_singleb, building_lifetime, retrofit_plan)
        rec = function_make_input_vector_and_predict_over_time(idf, building, records_init_singleb, building_lifetime, retrofit_plan)[-1]
        # print("OUTPUT:", recs)

        # recs = function_make_input_vector_and_predict(idf, building, records_init_singleb)[0]  # For now, just take the first.
        heating = rec['heating_dem_m2']
        cooling = rec['cooling_dem_m2']
        gas = rec["heating_gas_m2"]
        elec = rec["heating_elec_m2"]
        co2 = rec["carbon_emissions"]

        # Unify this somehow... We either use zero-prefixed strings, or we use integers
        fixed_bag_id = f"{bag_id:0>16}"
        original: dict[str, float] | None = base_prediction_cache.get(fixed_bag_id, None)

        # if original and "cooling_demand" in original and "heating_demand" in original:
        #     res[bag_id] |= {"delta": {"heating_demand": heating-original["heating_demand"], "cooling_demand": cooling-original["cooling_demand"]}}

        preds = {"heat": heating, "cool":cooling, "elec": elec,"gas": gas,"co2": co2}
        retr = {"retrofit" : {"passive": req.r_package, "active": [x.model_dump() for x in req.r_active]}}
        user_cache(user_id)[bag_id] = preds | retr
        res[bag_id] = preds | retr

    return json.dumps(res)


@app.get("/api/idf/")
def get_idf_image(user_id: str, bag_id: int):
    n_sim=2
    (idf, building, records_init_singleb) = get_initialized_idf(bag_id, n_sim)
    idf.view_model(test=True)

    f = io.BytesIO()
    plt.gcf().savefig(f, format = 'png')
    f.seek(0)
    return Response(content=f.read(), media_type="image/png")
