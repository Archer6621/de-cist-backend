from pathlib import Path

from diskcache import Cache

from retrpred.src.gen_function_make_geometry_idf_sample_initial import (
    make_geometry_idf_sample_distr,
)
from retrpred.src.ep_function_make_enriched_idf import make_enriched_idf

cache_root = Path('~/de-cist').expanduser()
cache_root.mkdir(exist_ok=True, parents=True)
building_cache = Cache(cache_root / 'buildings')

@building_cache.memoize()
def get_idf(bag_id: int, n_sim: int):
    (idf, building, records_init_singleb) = make_geometry_idf_sample_distr(f'["{bag_id:0>16}"]', n_sim)
    idf = make_enriched_idf(idf, building, records_init_singleb)
    return (idf, building, records_init_singleb)


def get_initialized_idf(bag_id: int, n_sim: int):
    (idf, building, records_init_singleb) = get_idf(bag_id, n_sim)
    idf.read()  # NOTE: This is needed for geomeppy to properly maintain its epp indices, since these are class attributes.
    return (idf, building, records_init_singleb)
