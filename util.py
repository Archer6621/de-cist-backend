from pathlib import Path

from diskcache import Cache

from retrpred.src.gen_function_make_geometry_idf_sample_initial import (
    make_geometry_idf_sample_distr,
)
from retrpred.src.ep_function_make_enriched_idf import make_enriched_idf
from retrpred.src.niemann import nieman_data


cache_root = Path('~/de-cist').expanduser()
cache_root.mkdir(exist_ok=True, parents=True)
building_cache = Cache(cache_root / 'buildings')
niemann_cache = Cache(cache_root / 'niemann')
pure_building_cache = Cache(cache_root / 'pure_buildings')

@niemann_cache.memoize()
def get_niemann(bag_id: int):
    bag_pand_id = f'["{bag_id:0>16}"]'
    building_data = nieman_data[nieman_data['BAG Pand ID'] == bag_pand_id] # querying the data for the specific building unit
    bag_adres_id = building_data['BAG Adres ID'].values[0] if not building_data.empty else None     
    building_data = nieman_data[nieman_data['BAG Adres ID'] == bag_adres_id] # querying the data for the specific building unit
    if len(building_data.index):
        return building_data.iloc[0]
    return None

@building_cache.memoize()
def get_idf(bag_id: int, n_sim: int):
    (idf, building, records_init_singleb) = make_geometry_idf_sample_distr(f'["{bag_id:0>16}"]', n_sim)
    idf = make_enriched_idf(idf, building, records_init_singleb)
    pure_building_cache[bag_id] = building
    return (idf, building, records_init_singleb)


def get_initialized_idf(bag_id: int, n_sim: int):
    (idf, building, records_init_singleb) = get_idf(bag_id, n_sim)
    idf.read()  # NOTE: This is needed for geomeppy to properly maintain its epp indices, since these are class attributes.
    return (idf, building, records_init_singleb)
