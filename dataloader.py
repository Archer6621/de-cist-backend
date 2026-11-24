from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_data(data_path='./data'):
	dfs: list[gpd.GeoDataFrame] = []
	for p in Path(data_path).glob('*.shp'):
		print("Loading shapefile data:", p)
		sub_gdf: gpd.GeoDataFrame = gpd.read_file(p)
		sub_gdf.crs = "EPSG:28992"
		sub_gdf["identifica"] = sub_gdf["identifica"].apply(lambda x: x.split('.')[-1]).astype('int64')
		dfs.append(sub_gdf.to_crs("EPSG:4326"))
	print("TOTAL DATAFRAMES LOADED:", len(dfs))
	return pd.concat(dfs).set_index('identifica')
