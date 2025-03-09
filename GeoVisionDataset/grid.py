
import pandas as pd
import math
import numpy as np
import shapely
import os
import geopandas as gpd

def tile_size_from_bounds(bounds:gpd.GeoSeries):
    minx,miny,maxx,maxy = bounds.to_crs(
                        bounds.estimate_utm_crs()
                    ).total_bounds

    tile_size = (
        (maxx - minx),
        (maxy - miny)
    )
    return tile_size

def ipyleaflet_drawable_map(center=[0, 0], zoom=11, height="800px"):
    import ipyleaflet
    import geopandas as gpd
    from ipyleaflet import DrawControl, Map
    from shapely.geometry import shape
    from IPython.display import display
    
    """
    Creates an interactive ipyleaflet map with drawing controls (squares & polygons).
    
    Returns:
        - m: The ipyleaflet map object
        - get_drawn_geometries: Function to return stored geometries as a GeoDataFrame
    """
    # Create a map
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, layout={'height': height})

    google_hybrid = ipyleaflet.TileLayer(
        url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        name="Google Hybrid",
        attribution="Google"
    )
    m.add_layer(google_hybrid)

    # Create a DrawControl object with squares (rectangles) and polygons
    draw_control = DrawControl(
        rectangle={"shapeOptions": {"color": "blue"}},  # Allow squares & rectangles
        polygon={"shapeOptions": {"color": "blue"}},    # Allow polygons
        circle={},
        polyline={},
        marker={},
        circlemarker={}
    )

    # Add the DrawControl to the map
    m.add_control(draw_control)

    # Initialize an empty list to store drawn geometries
    drawn_geometries = []

    # Handle drawn geometries
    def handle_draw(self, action, geo_json):
        """Callback function to store drawn geometries."""
        if action == 'created':  # Store new geometries only
            geometry = shape(geo_json['geometry'])  # Convert GeoJSON to Shapely geometry
            drawn_geometries.append(geometry)
            print(f"New geometry added: {geometry}")

    # Register the handle_draw function
    draw_control.on_draw(handle_draw)

    def get_drawn_geometries():
        """Returns the drawn geometries as a GeoDataFrame."""
        if drawn_geometries:
            gdf = gpd.GeoDataFrame({"geometry": drawn_geometries}, crs="EPSG:4326")
            return gdf
        else:
            return None  # No geometries drawn yet
    
    return m, get_drawn_geometries
    

def user_geom_leaflet_map(m):
    s = m.to_html()
    geoms = []
    idx = 0
    while True:
        start = s.find('"geometry":')
        if start == -1:
            break

        idx += 1
        end = s[start:].find("}")
        ss='{ "type": "Feature", '+s[start:start+end]+ '} }\n'
        geoms.append(ss)
        s = s[start+end:]

    geojson_str = '''{\n
        "type": "FeatureCollection",\n
        "features": [\n'''
    for i in geoms:
        geojson_str = geojson_str + i + ","

    geojson_str = geojson_str[0:-1]
    geojson_str = geojson_str + ']}'
    g=gpd.read_file(geojson_str,driver="GeoJSON")
    g.crs = 4326
    return gpd.GeoSeries(g.geometry.union_all(),crs=g.crs)


class Grid:
    def __init__(self,dataset_bounds:gpd.GeoSeries, tile_size:tuple=None, resolution:tuple=None, shape:tuple=None, 
                grid_bounds:gpd.GeoSeries=None, tile_in_dataset:float=0, 
                overlap:float=0): 
        from shapely import box

        if (type(tile_size) == tuple) and (type(resolution) == tuple):
            if type(shape) == tuple:
                raise Exception("Too many arguments (shape, resolution and tile_size). One of them should be set to None.")

            self.shape = (int(tile_size[0] / resolution[0]), int(tile_size[1] / resolution[1]))
            self.tile_size = tile_size
            self.resolution = resolution
        elif tile_size is None:
            if shape is None:
                raise Exception("shape argument not set.")
            elif resolution is None:
                raise Exception("Set either resolution or tile_size.")
            
            self.tile_size = (shape[0] * resolution[0], shape[1] * resolution[1])
            self.resolution = resolution
            self.shape = (int(shape[0]),int(shape[1]))
        else:
            self.tile_size = tile_size
            self.resolution = (tile_size[0] / shape[0], tile_size[1] / shape[1])
            self.shape = (int(shape[0]),int(shape[1]))

        if dataset_bounds.crs.is_projected == True:
            self.proj_crs = dataset_bounds.crs 
        elif (grid_bounds is not None) and grid_bounds.crs.is_projected == True:
            self.proj_crs = grid_bounds.crs 
        else:
            self.proj_crs = dataset_bounds.estimate_utm_crs(datum_name='WGS 84')

        self.dataset_bounds = dataset_bounds.to_crs(self.proj_crs)
        if grid_bounds is None:
            self.grid_bounds = self.dataset_bounds.total_bounds
        else:
            self.grid_bounds = grid_bounds.to_crs(self.proj_crs).total_bounds

        overlap = (self.tile_size[0] * overlap, self.tile_size[1] * overlap)
        overlap = (overlap[0] + resolution[0] - overlap[0] % resolution[0], overlap[1] + resolution[1] - overlap[1] % resolution[1])
        self.centroid_dist = (self.tile_size[0] - overlap[0], self.tile_size[1] - overlap[1])
        self.overlap = overlap

        grid_size = (int(math.ceil((self.grid_bounds[2] - self.grid_bounds[0]) / self.centroid_dist[0])),
                          int(math.ceil((self.grid_bounds[3] - self.grid_bounds[1]) / self.centroid_dist[1])))

        minx = np.arange(self.grid_bounds[0],self.grid_bounds[2],self.centroid_dist[0]) 
        miny = np.arange(self.grid_bounds[1],self.grid_bounds[3],self.centroid_dist[1])
        centroid_x = np.arange(self.grid_bounds[0]+self.overlap[0]+self.centroid_dist[0]/2,
                                    self.grid_bounds[2]+self.overlap[0]+self.centroid_dist[0]/2,
                                    self.centroid_dist[0])
        centroid_y = np.arange(self.grid_bounds[1]+self.overlap[1]+self.centroid_dist[1]/2,
                                    self.grid_bounds[3]+self.overlap[1]+self.centroid_dist[1]/2,
                                    self.centroid_dist[1])

        buffer = np.sqrt(2)*max(self.centroid_dist[0],self.centroid_dist[1])
        buffer = -buffer + (1 - tile_in_dataset/1) * buffer * 2
        dataset_bounds_with_buffer = self.dataset_bounds.buffer(buffer,resolution=4).union_all()
        valid_inds = []
        for g in shapely.get_parts(dataset_bounds_with_buffer):
            iis,jjs = self.__bounds_to_ij(g,grid_size)

            i = np.repeat(iis,len(jjs))
            j = np.tile(jjs,len(iis))

            t = shapely.contains_xy(dataset_bounds_with_buffer,centroid_x[i],centroid_y[j])
            inds = i[t] + j[t] * grid_size[0]
            valid_inds += inds.tolist()


        if len(valid_inds) == 0:
            raise Exception("No tiles are inside the bounds you gave")

        self.grid = pd.concat([gpd.GeoSeries([box(*self.__calc_bounds(minx,miny,i,grid_size))],crs=self.proj_crs) for i in valid_inds])
        if (tile_in_dataset > 0) and (tile_in_dataset <= 1):
            self.grid = self.grid[self.grid.geometry.intersection(self.dataset_bounds.to_crs(self.grid.crs).union_all()).area / self.grid.area >= tile_in_dataset/1]
        
        self.grid = self.grid.reset_index(drop=True)
        self.grid_boundary = gpd.GeoSeries(self.grid.union_all().boundary)

    def __bounds_to_ij(self,dataset_bounds,grid_size):
        bounds = dataset_bounds.bounds
        min_i = int(math.floor((bounds[0] - self.grid_bounds[0]) / self.centroid_dist[0]))
        min_j = int(math.floor((bounds[1] - self.grid_bounds[1]) / self.centroid_dist[1])) 
        if min_i < 0:
            min_i = 0
        
        if min_j < 0:
            min_j = 0

        max_i = int(math.ceil((bounds[2] - self.grid_bounds[0]) / self.centroid_dist[0])) 
        max_j = int(math.ceil((bounds[3] - self.grid_bounds[1]) / self.centroid_dist[1])) 
        if max_i > grid_size[0]:
            max_i = grid_size[0]

        if max_j > grid_size[1]:
            max_j = grid_size[1]

        return (range(min_i,max_i),range(min_j,max_j))

    
    def __calc_bounds(self,minx,miny,index,grid_size):
        j = int(math.floor(index/grid_size[0]))
        i = int(index % grid_size[0])

        minx = minx[i]
        miny = miny[j]
        maxx = minx + self.centroid_dist[0]
        maxy = miny + self.centroid_dist[1]
        minx -= self.overlap[0]
        maxx += self.overlap[0]
        miny -= self.overlap[1]
        maxy += self.overlap[1]

        bounds = [minx,miny,maxx,maxy]

        return bounds
    
    def geometry_to_index(self,x,y=None,crs=None):
        if y is not None:
            if type(x) is int or type(x) is float: 
                x = [x]

            if type(y) is int or type(y) is float: 
                y = [y]

            if len(x) != len(y):
                raise Exception(f"x {len(x)} and y {len(y)} must have the same lengths.")
            
            if crs is None:
                raise Exception("crs should be set but got None.")
            
            x = gpd.points_from_xy(x,y,crs=crs)

        if type(x) is gpd.GeoDataFrame:
            x = x.geometry 

        if type(x) != gpd.GeoSeries: 
            raise Exception(f"Geometry type should be geopandas.GeoSeries but got {type(x)}")
        
        indices = self.grid.index[self.grid.intersects(x.to_crs(self.proj_crs))]
        return list(indices)
    
    def save_metadata(self,path):
        grid_bounds = gpd.GeoSeries([shapely.geometry.box(*self.grid_bounds)],crs=self.proj_crs)
        dataset_bounds = self.dataset_bounds.to_crs(self.proj_crs)
        grid = self.grid.to_crs(self.proj_crs)

        file = os.path.normpath(path+"/grid_tiles.geojson")
        grid.to_file(file,driver="GeoJSON")
        print(f"Grid tiles saved as {file}") 

        file = os.path.normpath(path+"/grid_bounds.geojson")
        grid_bounds.to_file(file,driver="GeoJSON")
        print(f"Grid bounds saved as {file}")  

        file = os.path.normpath(path+"/dataset_bounds.geojson")
        dataset_bounds.to_file(file,driver="GeoJSON")

        print(f"Dataset bounds saved as {file}")      
