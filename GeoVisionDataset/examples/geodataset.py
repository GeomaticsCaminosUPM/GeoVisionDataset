import warnings 
import os
import numpy as np
import geopandas as gpd 
import matplotlib.pyplot as plt
import shapely

from .grid import Grid
from .image import ImageData
from .segmentation import SegmentationData 

"""TODO: 
Ipyleaflet basemaps
contours in coco do not show holes. 
Add a test dataset option to not save annotation.
Maybe add segmentation providers for all annotation formats
Every log iterations save coco anns so that you can recover them if the download stops
Allow to load coco anns and add new ones. Implement coco anns with overwrite = False
"""

def add_basemap(m,name='Google Satellite Hybrid',transparent=False):
    import folium
    if type(name) is str:
        basemaps = {
            'Google Maps': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Maps',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Google Satellite': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Google Terrain': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Terrain',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Google Satellite Hybrid': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True,
                transparent = transparent
            ),
            'Esri Satellite': folium.TileLayer(
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr = 'Esri',
                name = 'Esri Satellite',
                overlay = True,
                control = True,
                transparent = transparent
            )
        }
        basemaps[name].add_to(m)
    else:
        if "ipyleaflet.leaflet.WMSLayer" in str(type(name)):
            folium.raster_layers.WmsTileLayer(url = name.url,
                            layers = name.layers,
                            transparent = transparent, 
                            fmt="image/jpeg",
                            name = 'Background',
                            ).add_to(m)
        else:
            name.add_to(m)
    return None

def plot_coco(img,anns, instances:bool=True, n_classes=50):
    from pycocotools import mask as mask_tools
    plt.imshow(img)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    if len(anns) == 0:
        return None

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    mask = np.zeros((sorted_anns[0]['segmentation']['size'][0], sorted_anns[0]['segmentation']['size'][1], 3), dtype=np.uint8)

    # Generate a list of n colors from the `gist_rainbow` colormap
    cmap = plt.colormaps['gist_rainbow']
    colors = (cmap(np.linspace(0, 1, n_classes))[:, :3]*255).astype(np.uint8)  # Convert to RGB

    for ann in sorted_anns:
        m = mask_tools.decode(ann['segmentation'])
        color_mask = colors[ann['category_id']-1]
        mask[m == 1] = color_mask
        if instances is True:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(mask, contours, -1, (255,255,255), thickness=1) 

    alpha_channel = np.ones((mask.shape[0],mask.shape[1],1),dtype=mask.dtype) * 127
    alpha_channel[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)] = 255 
    alpha_channel[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 0 
    mask = np.concatenate((mask, alpha_channel),axis=2)
    mask[:,:,:3][mask[:,:,3] == 255] = 0
 
    ax.imshow(mask)
    
    return None

def plot_raster(img,anns_semantic,anns_instances=None, n_classes=50):
    plt.imshow(img)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    mask = np.zeros((anns_semantic.shape[0], anns_semantic.shape[1], 3), dtype=np.uint8)
    # Generate a list of n colors from the `gist_rainbow` colormap
    cmap = plt.colormaps['gist_rainbow']
    colors = (cmap(np.linspace(0, 1, n_classes))[:, :3]*255).astype(np.uint8)  # Convert to RGB

    for id in np.unique(anns_semantic):
        if id == 0:
            continue 

        color_mask = colors[id-1]
        mask[anns_semantic == id] = color_mask

    if anns_instances is not None:
        for id in np.unique(anns_instances):
            if id == 0:
                continue 

            import cv2
            contours, _ = cv2.findContours((anns_instances == id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(mask, contours, -1, (255,255,255), thickness=1) 

    alpha_channel = np.ones((mask.shape[0],mask.shape[1],1),dtype=mask.dtype) * 127
    alpha_channel[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)] = 255 
    alpha_channel[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 0 
    mask = np.concatenate((mask, alpha_channel),axis=2)
    mask[:,:,:3][mask[:,:,3] == 255] = 0
 
    ax.imshow(mask)

    return None

def plot_maskformer(img,maskformer,n_classes=50):
    plot_raster(img,maskformer[:,:,0],maskformer[:,:,1],n_classes=n_classes)

def plot_gdf(img,gdf,bounds,n_classes=50):
    xmin,ymin,xmax,ymax=bounds.total_bounds
    plt.imshow(img,extent=(xmin, xmax, ymin, ymax))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    gdf.plot(cmap='gist_rainbow',column='semantic_class',ax=ax,alpha=0.5,vmin=1,vmax=n_classes)
    gdf.geometry.boundary.plot(color='black',ax=ax)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    return None

class GeoDataset:
    def __init__(self,grid:Grid,image,segmentation=None,n_classes=None,hide_outside:bool=False,
                    min_area:float=0,min_object_coverage:float=0,min_tile_coverage:float=0,instances:bool=True) -> None:

        self.min_area = min_area 
        self.min_object_coverage = min_object_coverage 
        self.min_tile_coverage = min_tile_coverage 
        self.instances = instances

        self.ImageDataset = ImageData(image) 
        
        if segmentation is None:
            self.SegDataset = None 
        else:
            if n_classes is None:
                raise Exception("Please input the number of semantic classes at the n_classes keyword.")

            self.SegDataset = SegmentationData(segmentation,background_index=0,all_touched=True)
            self.n_classes = n_classes

        self.dataset_bounds = grid.dataset_bounds 
        self.grid_bounds = gpd.GeoSeries(shapely.box(*grid.grid_bounds),crs=grid.proj_crs)

        self.crs = self.ImageDataset.crs 
        self.grid = grid.grid.to_crs(self.crs)
        if len(self.grid) > 0:
            self.grid = gpd.GeoSeries(
                self.grid.bounds.apply(
                    lambda row: shapely.geometry.box(
                        row['minx'],
                        row['miny'],
                        row['maxx'],
                        row['maxy']
                    ),
                    axis=1
                ),
                crs=self.grid.crs
            )

        self.shape = grid.shape 
        self.tile_size = grid.tile_size 
        self.resolution = grid.resolution 

        self.hide_outside = hide_outside
        if hide_outside is True:
            self.hide_outside_bounds = self.dataset_bounds
        else:
            self.hide_outside_bounds = None

        if self.SegDataset is not None:
            self.background_index = self.SegDataset.background_index
        else:
            self.background_index = 0

        self.download_tiles = []

    def get_image(self,tile:int=None,bounds:gpd.GeoSeries|gpd.GeoDataFrame=None):
        if bounds is None:
            bounds = self.grid.geometry[tile:tile+1].reset_index(drop=True).copy()
            
        img, bounds = self.ImageDataset.img(bounds,self.shape,dataset_bounds=self.hide_outside_bounds)
        return img, bounds

    def get_annotation(self,tile:int=None,bounds:gpd.GeoSeries|gpd.GeoDataFrame=None,ann_mode='raster',instances:bool=None,
                                            min_area:float=None,min_object_coverage:float=None,min_tile_coverage:float=None):
        from . import segmentation 

        if min_area is None:
            min_area = self.min_area

        if min_object_coverage is None:
            min_object_coverage = self.min_object_coverage

        if min_tile_coverage is None:
            min_tile_coverage = self.min_tile_coverage

        if instances is None:
            instances = self.instances

        if bounds is None:
            if tile is None:
                raise Exception("Please set tile or bounds args.")

            bounds = self.grid.geometry[tile:tile+1].reset_index(drop=True).copy()

        geodataframe_ann = self.SegDataset.gdf_ann(bounds,dataset_bounds=self.hide_outside_bounds,
                            min_area=min_area, min_object_coverage=min_object_coverage,min_tile_coverage=min_tile_coverage)

        if not instances:
            geodataframe_ann = geodataframe_ann.dissolve(by='semantic_class').reset_index()
        
        if ann_mode == 'coco':
            instance_ids = np.arange(len(geodataframe_ann))
            semantic_class = list(geodataframe_ann['semantic_class'])
            raster_ann_instances = segmentation.gdf_to_raster_ann_instances(geodataframe_ann,self.shape,bounds,background_index=self.background_index) 
            anns = segmentation.raster_to_coco_ann(raster_ann_instances,semantic_class,instance_ids)
            return anns, bounds

        elif ann_mode == 'geodataframe':
            return geodataframe_ann, bounds

        elif ann_mode == 'raster':
            semantic_raster = segmentation.gdf_to_raster_ann_semantic(geodataframe_ann,self.shape,bounds,background_index=self.background_index)
            if instances:
                instance_raster = segmentation.gdf_to_raster_ann_instances(geodataframe_ann,self.shape,bounds,background_index=self.background_index)
                return (semantic_raster, instance_raster), bounds
            else:
                return semantic_raster, bounds

        elif ann_mode == 'maskformer':
            maskformer = segmentation.gdf_to_maskformer_ann(geodataframe_ann,self.shape,bounds,background_index=self.background_index)
            return maskformer, bounds

        else:
            raise Exception(f"Annotation mode {ann_mode} not implemented")

    def plot(self,image,annotation=None,ann_mode='raster', instances:bool=None,
            min_area:float=None,min_object_coverage:float=None,min_tile_coverage:float=None):

        if min_area is None:
            min_area = self.min_area

        if min_object_coverage is None:
            min_object_coverage = self.min_object_coverage

        if min_tile_coverage is None:
            min_tile_coverage = self.min_tile_coverage

        if instances is None:
            instances = self.instances

        if type(image) == int:
            image, bounds = self.get_image(image)
            annotation, _ = self.get_annotation(
                bounds=bounds,
                ann_mode=ann_mode,
                min_area=min_area,
                min_object_coverage=min_object_coverage,
                min_tile_coverage=min_tile_coverage,
                instances=instances
            )
        elif type(image) == gpd.GeoSeries:
            image, bounds = self.get_image(bounds=image)
            annotation, _ = self.get_annotation(
                bounds=bounds,
                ann_mode=ann_mode,
                min_area=min_area,
                min_object_coverage=min_object_coverage,
                min_tile_coverage=min_tile_coverage,
                instances=instances
            )   

        plt.figure(figsize=(10, 10))
        if annotation is not None:
            if ann_mode == 'coco':
                plot_coco(image,annotation,n_classes=self.n_classes)
            elif ann_mode == 'raster':
                if instances:
                    plot_raster(image,annotation[0],annotation[1],n_classes=self.n_classes)
                else:
                    plot_raster(image,annotation,n_classes=self.n_classes)

            elif ann_mode == 'masformer':
                plot_maskformer(image,annotation,n_classes=self.n_classes)
            elif ann_mode == 'geodataframe':
                plot_gdf(image,annotation,n_classes=self.n_classes)
            else:
                raise Exception(f"Annotation mode {ann_mode} not implemented")
        else:
            plt.imshow(image)

        plt.axis('off')
        plt.show() 
        return None

    def to_map(self,bounds=None,mode:str='grid',basemap='Google Satellite Hybrid',show_tile_id:bool=True,m=None):
        import folium

        if bounds is None:
            bounds = self.dataset_bounds
        elif type(bounds) == list:
            if type(bounds[0]) != int:
                raise Exception("Please input a list of ints with the grid indices to plot.")

            bounds = self.grid.geometry[bounds]
        elif type(bounds) == int: 
            bounds = self.grid.geometry[bounds:bounds+1] 

        if (self.SegDataset is not None) and (mode != 'grid') and (mode != 'image'):
            geometry = self.SegDataset.geometry(bounds=bounds)
            if len(geometry) > 0:
                if geometry is not None:
                    if type(geometry) is gpd.GeoSeries:
                        geometry = gpd.GeoDataFrame({'id':geometry.index},geometry=geometry.geometry,crs=geometry.crs)
                    else:
                        geometry['id'] = geometry.index

                    if 'semantic_class' in geometry.columns:
                        m=geometry[geometry['semantic_class'] >= 1].explore(m=m,column='semantic_class',cmap='gist_rainbow',legend=False,vmin=1,vmax=self.n_classes)
                        m=geometry[geometry['semantic_class'] == 0].explore(m=m,color='black',legend=False)
                    else:
                        m=geometry.explore(m=m,column='id',cmap='prism',legend=False)

        should_add_basemap = m is None
        m = self.dataset_bounds.to_crs(4326).explore(
            m=m,
            style_kwds={"color": "black", "stroke": True, "fill": False, "weight": 5}
        )
        grid = gpd.GeoDataFrame({'id':np.random.rand(len(self.grid))},geometry=self.grid.geometry.copy(),crs=self.grid.crs)
        grid = grid[grid.intersects(bounds.to_crs(grid.crs).union_all())]
        m = grid.to_crs(4326).explore(
            m=m,
            style_kwds={"color":"black","stroke":True,"fill":False}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centroid_x = grid.geometry.centroid.to_crs(4326).x
            centroid_y = grid.geometry.centroid.to_crs(4326).y

        if show_tile_id is True:
            for i in grid.index:
                text = str(i)
                #center = centroids[i:i+1].union_all().coords[0]
                center = [centroid_y[i],centroid_x[i]]
                style = "font-size: 12px; font-weight: bold; color: rgba(255, 255, 255, 0.5); text-align: center; text-shadow: -2px -2px 2px black, 2px -2px 2px black, -2px 2px 2px black, 2px 2px 2px black;"
                folium.Marker(location=center, icon=folium.DivIcon(html=f"<div style='{style}'>{text}</div>")).add_to(m)

        if should_add_basemap is True:
            add_basemap(m,basemap)

        if mode == 'grid':
            return m

        if self.ImageDataset is not None:
            basemap = self.ImageDataset.basemap(bounds=bounds)
            if basemap is not None:
                add_basemap(m,basemap)

        return m 

    def select_all_tiles(self):
        self.download_tiles = list(self.grid.index)

    def select_random_tiles(self,n:int):
        import random
        self.download_tiles = random.sample(list(self.grid.index), n)

    def select_tiles(self,ids:int|list):
        if type(ids) == int: 
            ids = [ids]

        self.download_tiles = ids   

    def select_tiles_in_bounds(self,bounds:gpd.GeoSeries,how='intersects'):
        if how == 'intersects':
            self.download_tiles = self.grid.index[self.grid.intersects(bounds.to_crs(self.grid.crs).union_all())]
        elif how == 'conatins':
            self.download_tiles = self.grid.index[shapely.contains(bounds.to_crs(self.grid.crs).union_all(),self.grid.geometry)]
        else: 
            raise Exception(f"{how} not implemented")

    def download(self,img_path:str,anns_path:str=None,ann_mode='coco',log:int=10,on_errors:str='ignore',overwrite:bool=False,allow_empty_anns:bool=False):
        import json
        from . import raster as rasterlib

        if ann_mode == 'coco':
            overwrite = True #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if self.ImageDataset is not None:     
            if not os.path.isdir(img_path):
                print(f"Creating image download path {img_path}")
                os.makedirs(img_path)   

        if self.SegDataset is not None: 
            if anns_path is None:
                anns_path = img_path 

            if not os.path.isdir(anns_path):
                print(f"Creating annotations download path {anns_path}")
                os.makedirs(anns_path)

        n = 0
        coco_imgs = []
        coco_anns = []
        all_labels = []

        num_iterations = len(self.download_tiles)

        for i in self.download_tiles:
            n += 1
            # Print the status bar every log iterations
            if n % log == 0 or n == num_iterations:  # Print on every log iteration or the last one
                completed = (n / num_iterations) * 100  # Calculate completion percentage
                print(f"\rProgress: [{'#' * (n // log)}{'.' * ((num_iterations - n) // log)}] {completed:.0f}% ({n}/{num_iterations})", end="")
                
            bounds = self.grid.geometry[i:i+1].reset_index(drop=True).copy()
            if self.ImageDataset is not None:
                img_file = os.path.normpath(img_path+f"/tile_{i}.jpg")
                
                if overwrite or not(os.path.isfile(img_file)):
                    try: 
                        img, bounds = self.get_image(i,bounds=bounds)
                    except Exception as e:
                        if on_errors == "stop":
                            raise e 
                        else: 
                            warnings.warn(f"Skipping tile {i} due to error on image: {e}", UserWarning)                      
                else:
                    bounds = rasterlib.bounds(img_file)


            if self.SegDataset is not None:
                if ann_mode == 'coco': 
                    ann_file = os.path.normpath(anns_path + "/annotations_coco.json")
                elif ann_mode == 'geodataframe':
                    ann_file = os.path.normpath(anns_path + f"/tile_{i}.gpkg")
                elif ann_mode == 'raster': 
                    ann_file = os.path.normpath(anns_path + f"/semantic/tile_{i}_semantic.png")
                    ann_file_semantic = ann_file
                    ann_file_instances = os.path.normpath(anns_path + f"/instances/tile_{i}_instances.png")
                elif ann_mode == 'maskformer':
                    ann_file = os.path.normpath(anns_path + f"/tile_{i}.png")
                else:
                    raise Exception(f"Annotation mode {ann_mode} not implemented")
                
                if overwrite or not(os.path.isfile(ann_file)):
                    annotation = None 
                    try: 
                        annotation, _ = self.get_annotation(
                            i,
                            bounds=bounds,
                            ann_mode=ann_mode,
                            min_area=self.min_area,
                            min_object_coverage=self.min_object_coverage,
                            min_tile_coverage=self.min_tile_coverage,
                            instances=self.instances
                        ) 

                    except Exception as e: 
                        warnings.warn(f"Skipping tile {i} due to error on annotation: {e}", UserWarning)   
                        continue
                        
                    if ann_mode == 'coco':
                        if len(annotation) > 0:
                            instance_ids = np.arange(len(coco_anns),len(coco_anns)+len(annotation))
                            semantic_classes = []
                            for i in range(len(annotation)):
                                annotation[i]['id'] = int(instance_ids[i])
                                semantic_classes.append(int(annotation[i]['category_id']))

                            coco_anns += annotation
                        elif allow_empty_anns == False:
                            continue

                        image_dict = {'image_id':int(i),'file_name':str(img_path),'height':int(self.shape[0]),'width':int(self.shape[1])}
                        coco_imgs.append(image_dict)
                        all_labels = all_labels + list(np.unique(semantic_classes))

                    elif ann_mode == 'raster':
                        if (allow_empty_anns == False) and (np.max(annotation[0]) == 0):
                            continue 

                        if instances:
                            rasterlib.save(ann_file_semantic,annotation[0],bounds=bounds,driver='PNG')
                            rasterlib.save(ann_file_instances,annotation[1],bounds=bounds,driver='PNG')
                        else:
                            rasterlib.save(ann_file_semantic,annotation,bounds=bounds,driver='PNG')

                    elif ann_mode == 'maskformer':
                        if (allow_empty_anns == False) and (np.max(annotation) == 0):
                            continue 

                        annotation = np.transpose(annotation, (2, 0, 1))
                        rasterlib.save(ann_file,annotation,bounds=bounds,driver='PNG')
                    elif ann_mode == 'geodataframe':
                        if (allow_empty_anns == False) and (len(annotation) == 0):
                            continue 

                        annotation.to_file(ann_file)

                
            if self.ImageDataset is not None:
                if overwrite or not(os.path.isfile(img_file)):
                    img = np.transpose(img, (2, 0, 1))
                    rasterlib.save(img_file,img,bounds=bounds,driver='JPEG')

        if ann_mode == 'coco':
            if overwrite or not(os.path.isfile(anns_path + "/annotations_coco.json")):
                json_dict = {
                    'images' : coco_imgs,
                    'annotation' : coco_anns,
                    'categories' : [{'id' : int(i)} for i in np.unique(all_labels)]
                }
                json_file = os.path.normpath(anns_path + "/annotations_coco.json")
                with open(json_file, "w") as file:
                    json.dump(json_dict, file, indent=4)  # `indent=4` for pretty formatting


        grid_bounds = self.grid_bounds
        dataset_bounds = self.dataset_bounds
        grid = self.grid

        file = os.path.normpath(img_path+"/grid_tiles.geojson")
        grid.to_file(file,driver="GeoJSON")
        print(f"Grid tiles saved as {file}") 

        file = os.path.normpath(img_path+"/grid_bounds.geojson")
        grid_bounds.to_file(file,driver="GeoJSON")
        print(f"Grid bounds saved as {file}")  

        file = os.path.normpath(img_path+"/dataset_bounds.geojson")
        dataset_bounds.to_file(file,driver="GeoJSON")

        print(f"Dataset bounds saved as {file}")  

        self.ImageDataset.save_metadata(img_path)

        if self.SegDataset is not None:
            if anns_path != img_path:
                grid_bounds = self.grid_bounds
                dataset_bounds = self.dataset_bounds
                grid = self.grid

                file = os.path.normpath(anns_path+"/grid_tiles.geojson")
                grid.to_file(file,driver="GeoJSON")
                print(f"Grid tiles saved as {file}") 

                file = os.path.normpath(anns_path+"/grid_bounds.geojson")
                grid_bounds.to_file(file,driver="GeoJSON")
                print(f"Grid bounds saved as {file}")  

                file = os.path.normpath(anns_path+"/dataset_bounds.geojson")
                dataset_bounds.to_file(file,driver="GeoJSON")

                print(f"Dataset bounds saved as {file}")  

            self.SegDataset.save_metadata(anns_path)              

        

     
