import geopandas as gpd 
import pandas as pd 

def overpass_api_query(query:str,bounds:gpd.GeoDataFrame|gpd.GeoSeries):
    import requests
    from osm2geojson import json2geojson

    bbox = bounds.to_crs(4326).total_bounds
    bbox = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = query.replace("{{bbox}}",bbox)
    query = query.replace("{bbox}",bbox)
    query = query.replace("[out:xml]","[out:json]")
    overpass_url = 'https://overpass-api.de/api/interpreter'
    response = requests.get(overpass_url, params={'data': query})
    if response.status_code != 200:
        raise Exception(f"Overpass turbo api query failed with statud code {response.status_code}. The failed query was: {query}")
        
    geojson_response = json2geojson(response.json())
    gdf = gpd.GeoDataFrame.from_features(geojson_response,crs=4326).reset_index(drop=True)
    new_gdf = gdf['tags'].apply(pd.Series)
    if 'type' in new_gdf.columns:
        new_gdf = new_gdf.rename(columns={'type':'geometry_type'})
    
    gdf = pd.concat([gdf.drop(columns=['tags']), new_gdf], axis=1).reset_index(drop=True)
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    return gdf.to_crs(bounds.crs)
