from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_data(data_path='./data'):
	dfs: list[gpd.GeoDataFrame] = []
	for p in Path(data_path).glob('*.shp'):
		sub_gdf: gpd.GeoDataFrame = gpd.read_file(p)
		sub_gdf.crs = "EPSG:28992"
		dfs.append(sub_gdf.to_crs("EPSG:4326"))
	return pd.concat(dfs).set_index('identifica')
