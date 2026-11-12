import sys
from pathlib import Path

import geopandas as gpd

# TODO: Can make this a bit nicer or more elaborate.

if len(sys.argv) > 1:
	path = Path(sys.argv[1])
	gdf = gpd.read_file(path)
	gdf.crs = "EPSG:28992"
	gdf.to_crs("EPSG:4326").to_file(path.with_name(path.stem + "_wgs84.geojson"), driver="geojson")
else:
	print("Please provide a path to a geojson file.")