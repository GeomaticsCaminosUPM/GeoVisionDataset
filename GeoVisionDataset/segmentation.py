import geopandas as gpd
import pandas as pd
from . import raster as rasterlib
import rasterio
import warnings
import numpy as np 
import os
import shapely

def buffer_in_m(geom,buffer):
    orig_crs = geom.crs
    if orig_crs.is_projected == False:
        utm = geom.to_crs(geom.estimate_utm_crs(datum_name='WGS 84'))
    
    utm = utm.buffer(buffer)
    return utm.to_crs(orig_crs)


def crop_gdf(G,g):
    G_is_geodataframe = False
    G = gpd.GeoDataFrame(geometry=G)
    g_is_geodataframe = False 
    g = gpd.GeoDataFrame(geometry=g)  
    g_crs = g.crs
    g = g.to_crs(G.crs) 
    resG = gpd.sjoin(G, g, how='inner', predicate='intersects')
    resG = resG.reset_index(drop=True)
    resG = resG.geometry
    resG = resG[resG.is_valid].reset_index(drop=True)
    return resG

def _coco_annotation(binary_mask,semantic_class,instance_id):
    from pycocotools import mask as mask_tools
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    encoded_mask = mask_tools.encode(fortran_mask)
    area = mask_tools.area(encoded_mask)
    bbox = mask_tools.toBbox(encoded_mask)
    annotation = {'id':int(instance_id),'category_id':int(semantic_class),'bbox':[float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])],'area':float(area),'iscrowd':0,
                    'segmentation':{'counts':encoded_mask['counts'].decode('utf-8'),'size':[int(encoded_mask["size"][0]),int(encoded_mask["size"][1])]}}
    
    return annotation 


def gdf_to_raster_ann_semantic(geodataframe_ann:gpd.GeoDataFrame,shape,bounds,background_index:int,all_touched:bool=False): 
    if type(shape) is int:
        bbox_utm = bounds.to_crs(bounds.estimate_utm_crs())
        dx = (bbox_utm.total_bounds[2] - bbox_utm.total_bounds[0])
        dy = (bbox_utm.total_bounds[3] - bbox_utm.total_bounds[1])
        if dx < dy:
            shape = (shape, int(dy/dx * shape))
        else:
            shape = (int(dx/dy * shape), shape)
    elif type(shape) is tuple:
        if len(shape) != 2:
            raise Exception(f"shape is wrong {shape}")
    else:
        raise Exception(f"shape should be int or tuple but got {type(shape)}")

    if "semantic_class" not in geodataframe_ann.columns:
        raise Exception(f"Column 'semantic_class' not in geodataframe_ann.")

    semantic_class = np.array(geodataframe_ann['semantic_class']).astype(int)

    if len(geodataframe_ann) == 0:
        return np.ones(shape,dtype=int) * background_index 
        

    raster,_ = rasterlib.rasterize(
        geodataframe_ann.geometry,
        shape,
        bounds,
        all_touched=all_touched,
        values=semantic_class,
        background_index=background_index
    ) 

    return raster

def gdf_to_raster_ann_instances(geodataframe_ann:gpd.GeoDataFrame,shape,bounds,background_index:int,all_touched:bool=False): 
    if type(shape) is int:
        bbox_utm = bounds.to_crs(bounds.estimate_utm_crs())
        dx = (bbox_utm.total_bounds[2] - bbox_utm.total_bounds[0])
        dy = (bbox_utm.total_bounds[3] - bbox_utm.total_bounds[1])
        if dx < dy:
            shape = (shape, int(dy/dx * shape))
        else:
            shape = (int(dx/dy * shape), shape)
    elif type(shape) is tuple:
        if len(shape) != 2:
            raise Exception(f"shape is wrong {shape}")
    else:
        raise Exception(f"shape should be int or tuple but got {type(shape)}")

    if "semantic_class" not in geodataframe_ann.columns:
        raise Exception(f"Column 'semantic_class' not in geodataframe_ann.")

    semantic_class = np.array(geodataframe_ann['semantic_class']).astype(int)
    instance_semantic_class = np.zeros(len(geodataframe_ann),dtype=int) 
    instance_semantic_class[semantic_class != background_index] = np.arange(1,np.sum(semantic_class != background_index)+1,dtype=int)

    if len(geodataframe_ann) == 0:
        return np.ones(shape,dtype=int) * background_index 

    raster,_ = rasterlib.rasterize(
        geodataframe_ann.geometry,
        shape,
        bounds,
        all_touched=all_touched,
        values=list(instance_semantic_class),
        background_index=background_index
    ) 

    return raster

def raster_to_coco_ann(ann_instance, semantic_class, instance_ids):
    #from torch.nn.functional import one_hot 
    #from torch import tensor, long

    # Number of classes (or maximum class label + 1)
    num_classes = np.max(ann_instance) + 1

    if num_classes <= 1:
        return [] 

    # Create one-hot encoded binary masks
    binary_masks = np.eye(num_classes, dtype=np.uint8)[ann_instance]

    #binary_masks = one_hot(tensor(ann_instance,dtype=long)).numpy()

    if (binary_masks.shape[2]-1) != len(semantic_class):
        raise Exception(f"Len of binary_masks is {(binary_masks.shape[2]-1)} but len of semantic_class is {len(semantic_class)}")

    if (binary_masks.shape[2]-1) != len(instance_ids):
        raise Exception(f"Len of binary_masks is {(binary_masks.shape[2])-1} but len of instance_ids is {len(instance_ids)}")

    annotations = [_coco_annotation(binary_masks[:,:,i+1],semantic_class[i],instance_ids[i]) for i in range(binary_masks.shape[2]-1)]
    
    return annotations

def gdf_to_maskformer_ann(geodataframe_ann:gpd.GeoDataFrame,shape,bounds,background_index:int):
    maskformer_ann = np.zeros((3,*shape),dtype=int)

    raster_ann_semantic = gdf_to_raster_ann_semantic(geodataframe_ann,shape,bounds,background_index)
    maskformer_ann[0,:,:] = raster_ann_semantic

    raster_ann_instances = gdf_to_raster_ann_instances(geodataframe_ann,shape,bounds,background_index)
    maskformer_ann[1,:,:] = raster_ann_instances
    
    return maskformer_ann




class SegmentationData:
    def __init__(self, ann_obj, background_index:int = 0, all_touched:bool = False) -> None:
        if type(ann_obj) != list:
            ann_obj = [ann_obj]

        for i in range(len(ann_obj)):
            if 'get' not in dir(ann_obj[i]): 
                raise Exception(f".get method not implemented in mask {i}")
            
            if 'geometry' not in dir(ann_obj[i]): 
                warnings.warn(f".geometry method not implemented in mask {i}", UserWarning)

            if 'save_metadata' not in dir(ann_obj[i]): 
                warnings.warn(f".save_metadata method not implemented in mask {i}", UserWarning)

        self.ann_obj = ann_obj
        self.background_index = background_index 
        self.all_touched = all_touched 

    def gdf_ann(self,bounds:gpd.GeoSeries,dataset_bounds:gpd.GeoSeries|gpd.GeoDataFrame=None,
                    min_area:float=0,min_object_coverage=0.05,min_tile_coverage=0.05,background_index:int=None):

        if background_index is None:
            background_index = self.background_index
        
        bounds = bounds.copy()
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

        geoms = pd.concat([i.get(bounds).to_crs(bounds.crs) for i in self.ann_obj])

        geoms = geoms[geoms.geometry.isna()==False].reset_index(drop=True)

        geoms = geoms[geoms.geometry.is_valid].reset_index(drop=True)

        geoms = geoms[geoms.geometry.is_empty == False].reset_index(drop=True)


        if len(geoms) > 0: 
            geoms_orig = geoms.copy()
            geoms.geometry = geoms.geometry.intersection(bounds.to_crs(geoms.crs).union_all())
            geoms = geoms[geoms.geometry.is_empty == False].reset_index(drop=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if dataset_bounds is not None:
                geoms.geometry = geoms.geometry.intersection(dataset_bounds.to_crs(geoms.crs).union_all())

            if len(geoms) > 0: 
                geoms = geoms[(geoms.geometry.area / geoms_orig.geometry.area) > min_object_coverage].reset_index(drop=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (min_area > 0) and (len(geoms) > 0):
                geoms = geoms[geoms.geometry.to_crs(geoms.estimate_utm_crs()).area > min_area].reset_index(drop=True)        

            if (min_tile_coverage > 0) and ((sum(geoms.geometry.area) / sum(bounds.to_crs(geoms.crs).area)) < min_tile_coverage):
                geoms = gpd.GeoDataFrame({'semantic_class':[]},geometry=[],crs=geoms.crs)

        if len(geoms) == 0: 
            geoms = gpd.GeoDataFrame({'semantic_class':[]},geometry=[],crs=geoms.crs)

        geoms = geoms.loc[geoms['semantic_class'] != background_index]

        return geoms

    def geometry(self,bounds:gpd.GeoDataFrame|gpd.GeoSeries):
        bounds = bounds.copy()
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

        geoms = []
        for i in self.ann_obj:
            if 'geometry' in dir(i):
                g = i.geometry(bounds)
                if type(g) is gpd.GeoDataFrame:
                    geoms.append(g.to_crs(bounds.crs))

        if len(geoms) == 0:
            return gpd.GeoDataFrame({'semantic_class':[]},geometry=[],crs=bounds.crs) 

        geoms = pd.concat(geoms)

        return geoms

    def save_metadata(self,path):
        for i in range(len(self.ann_obj)):
            if 'save_metadata' in dir(self.ann_obj[i]): 
                self.ann_obj[i].save_metadata(os.path.join(path,f"anns_metadata_{i}"))


# mask objects
    
class Polygon:
    def __init__(self, geoms:gpd.GeoDataFrame, semantic_class_column = None, label_transform_dict = None, 
                 all_touched=False, semantic_class:int=1):
        
        self.geoms = geoms
        crs = geoms.crs
        self.geoms.geometry = shapely.force_2d(self.geoms.geometry)
        self.geoms.crs = crs
        self.all_touched = all_touched
        self.semantic_class = semantic_class
        self.semantic_class_column = semantic_class_column 
        self.label_transform_dict = label_transform_dict

    def get(self,bounds:gpd.GeoSeries):
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

        bounds = bounds.to_crs(self.geoms.crs)
        vect_anns = self.geoms.intersection(bounds.union_all())
        vect_anns = vect_anns.explode(ignore_index=True,index_parts=False).reset_index(drop=True)
        vect_anns = vect_anns[vect_anns.geometry.is_empty == False].reset_index(drop=True)
        vect_anns = vect_anns[vect_anns.is_valid].reset_index(drop=True)

        if self.semantic_class_column is None:
            semantic_class = list(np.ones(len(vect_anns)) * self.semantic_class)  
        else:
            semantic_class = []
            for i in vect_anns[self.semantic_class_column]:
                if self.label_transform_dict is None:
                    c = i
                else:
                    try:
                        c = self.label_transform_dict[i] 
                    except:
                        c = self.semantic_class

                semantic_class.append(c)

        return gpd.GeoDataFrame({'semantic_class':semantic_class},geometry=vect_anns.geometry,crs=vect_anns.crs)
    
    def geometry(self,bounds:gpd.GeoSeries=None):
        if bounds is None:
            from shapely import box
            bounds = gpd.GeoSeries(box(*self.geoms.total_bounds),crs=self.geoms.crs)

        geoms = self.get(bounds)
        return geoms
    
    def save_metadata(self,path):
        path = path.split(".")
        path = f"{path[0]}_polygons.geojson"
        path = os.path.normpath(path)
        self.geoms.to_file(path)
        print(f"polygons saved as {path}")

class OSMPolygon:
    def __init__(self, overpass_query:str,
                 semantic_class_column:str = None, label_transform_dict:dict = None, 
                 all_touched:bool=False, semantic_class:int=1) -> None:

        self.overpass_query = overpass_query

        self.crs = 4326

        self.all_touched = all_touched

        self.semantic_class = semantic_class
        self.semantic_class_column = semantic_class_column 
        self.label_transform_dict = label_transform_dict

    def get(self,bounds):
        from . import osm
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)
        
        bounds = bounds.to_crs(self.crs)

        vect_anns = osm.overpass_api_query(self.overpass_query,bounds=bounds)
        vect_anns = vect_anns.loc[vect_anns.geometry.type == "Polygon"]

        crs = vect_anns.crs
        vect_anns.geometry = shapely.force_2d(vect_anns.geometry)
        vect_anns.crs = crs
        vect_anns.geometry = vect_anns.geometry.intersection(bounds.union_all())
        vect_anns = vect_anns.explode(ignore_index=True,index_parts=False).reset_index(drop=True)
        vect_anns = vect_anns[vect_anns.geometry.is_empty == False].reset_index(drop=True)
        vect_anns = vect_anns[vect_anns.is_valid].reset_index(drop=True)

        if self.semantic_class_column is None:
            semantic_class = list(np.ones(len(vect_anns)) * self.semantic_class)  
        else:
            semantic_class = []
            for i in vect_anns[self.semantic_class_column]:
                if self.label_transform_dict is None:
                    c = i
                else:
                    try:
                        c = self.label_transform_dict[i] 
                    except:
                        c = self.semantic_class

                semantic_class.append(c)

        return gpd.GeoDataFrame({'semantic_class':semantic_class},geometry=vect_anns.geometry,crs=vect_anns.crs)
    
    def geometry(self,bounds):
        bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs).to_crs(self.crs)
        geoms = self.get(bounds)
        return geoms

    def save_metadata(self,path):
        from . import osm
        path = path.split(".")
        path = os.path.normpath(path[0]+"_overpass_query.txt")
        with open(path, 'w') as file:
            file.write(self.overpass_query)

        print(f"osm overpass query saved as {path}")


class WFSPolygon:
    def __init__(self, wfs, typename:str=None,version:str=None, wfs_format:str=None,crs=4326, 
                 semantic_class_column:str = None, label_transform_dict:dict = None, 
                 all_touched:bool=False, semantic_class:int=1) -> None:
        from . import wms

        self.wfs, self.version = wms.build_wfs(wfs,version)
        self.typename = wms.check_wfs_typename(wfs,typename)

        self.wfs_format = wms.check_wfs_format(wfs=self.wfs,wfs_format=wfs_format,typename=self.typename,version=self.version)

        self.crs = wms.check_wfs_crs(self.wfs,typename=self.typename,crs=crs,version=self.version)

        self.all_touched = all_touched

        self.semantic_class = semantic_class
        self.semantic_class_column = semantic_class_column 
        self.label_transform_dict = label_transform_dict

    def get(self,bounds):
        from . import wms
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)
        
        bounds = bounds.to_crs(self.crs)
        vect_anns = wms.request_wfs_features(self.wfs, bounds = bounds, typename = self.typename, wfs_format=self.wfs_format, 
                                        version = self.version, crs = self.crs)

        crs = vect_anns.crs
        vect_anns.geometry = shapely.force_2d(vect_anns.geometry)
        vect_anns.crs = crs
        vect_anns.geometry = vect_anns.geometry.intersection(bounds.union_all())
        vect_anns = vect_anns.explode(ignore_index=True,index_parts=False).reset_index(drop=True)
        vect_anns = vect_anns[vect_anns.geometry.is_empty == False].reset_index(drop=True)
        vect_anns = vect_anns[vect_anns.is_valid].reset_index(drop=True)

        if self.semantic_class_column is None:
            semantic_class = list(np.ones(len(vect_anns)) * self.semantic_class)  
        else:
            semantic_class = []
            for i in vect_anns[self.semantic_class_column]:
                if self.label_transform_dict is None:
                    c = i
                else:
                    try:
                        c = self.label_transform_dict[i] 
                    except:
                        c = self.semantic_class

                semantic_class.append(c)

        return gpd.GeoDataFrame({'semantic_class':semantic_class},geometry=vect_anns.geometry,crs=vect_anns.crs)
    
    def geometry(self,bounds=None):
        from . import wms
        if bounds is None:
            bounds = wms.wfs_typename_bbox(self.wfs,typename=self.typename,version=self.version)
            bounds = gpd.GeoSeries(shapely.geometry.box(*bounds),crs=4326)
        
        bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs).to_crs(self.crs)
        geoms = self.get(bounds)
        return geoms

    def save_metadata(self,path):
        from . import wms
        path = path.split(".")
        path = os.path.normpath(path[0]+"_WFSCapabilities.xml")
        wms.save_wfs_getcapabilities(self.wfs,path)
        print(f"wfs capabilities saved as {path}")


class from_raster_files:
    def __init__(self,img_files, background_index, label_transform_dict = None, all_touched=False) -> None: 
        if type(img_files) is str:
            img_files = [img_files]

        self.crs= rasterlib.get_crs(img_files[0])

        img_bounds = {}
        for i in img_files:
            img_bounds[i] = rasterlib.bounds(i,crs=self.crs)
        
        self.img_files = img_files
        self.img_bounds = img_bounds
        self.label_transform_dict = label_transform_dict
        if label_transform_dict is not None:
            self.background_index = label_transform_dict[background_index] #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.label_transform_dict[255] = self.background_index
        else:
            self.background_index = background_index
            self.label_transform_dict = {}
            for i in range(255):
                self.label_transform_dict[i] = i 
            
            self.label_transform_dict[255] = self.background_index

        self.all_touched = all_touched

    def get(self,bounds:gpd.GeoSeries):
        from rasterio import features
        from shapely.geometry import shape

        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

        bounds = bounds.to_crs(self.crs)

        img_files = []
        for i in self.img_files:
            if bounds[0].intersects(self.img_bounds[i][0]):
                img_files.append(i)
        
        if len(img_files) == 0:
            print(f"No files found for bounds {str(bounds[0])}. Returning empty GeoSeries.")

            return gpd.GeoSeries([],crs=self.crs), None
        
        img,img_bounds,transform = rasterlib.merge(img_files,bounds=bounds)
        if self.label_transform_dict is not None:
            trans = np.vectorize(lambda x: self.label_transform_dict.get(x, x), otypes=[img.dtype])
            img = trans(img)
    
        img=img.astype(np.min_scalar_type(np.max(img)))

        mask = img != self.background_index
        shapes = features.shapes(img, mask=mask, transform=transform)

        _shapes = []
        semantic_class = []

        for g, v in shapes:
            _shapes.append(shape(g))
            semantic_class.append(v)

        if len(_shapes) == 0:
            gpd.GeoDataFrame({'semantic_class':[]},geometry=[],crs=shapes.crs)         
        
        shapes = gpd.GeoSeries(_shapes,crs=self.crs)
        return gpd.GeoDataFrame({'semantic_class':semantic_class},geometry=shapes.geometry,crs=shapes.crs)

    
    def geometry(self,bounds:gpd.GeoSeries=None):
        if bounds is None:
            bounds = gpd.GeoSeries(self.img_bounds.union_all(),crs=bounds.crs)
        else:
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

        geoms = self.get(bounds)
        return geoms
    
    def save_metadata(self,path):
        None

        
class LineString:
    def __init__(self,target_geoms:gpd.GeoSeries, other_geoms:gpd.GeoSeries=None, crop_area:gpd.GeoSeries=None, buffer:float=None,
                            snap_dist_far:float=0, snap_dist_close:float=0.1, grid_size:float=None, 
                            min_target_len:float=0, max_target_len:float=0, min_pol_area:float=0, max_pol_area:float=0, min_edges:int=5, max_edges:int=0, 
                            decimals:int=3, prefer_small_pols:bool=True, isolated_pols:bool=True, allow_overlap:bool=False, semantic_class:int=1):
    
        self.semantic_class = semantic_class

        self.decimals = decimals
        self.snap_dist_far = snap_dist_far
        self.snap_dist_close = snap_dist_close
        self.buffer = buffer
        self.grid_size = grid_size
        self.min_target_len = min_target_len
        self.max_target_len = max_target_len
        self.min_pol_area = min_pol_area
        self.max_pol_area = max_pol_area
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.prefer_small_pols = prefer_small_pols
        self.isolated_pols = isolated_pols
        self.allow_overlap = allow_overlap

        if crop_area is not None:
            crs = crop_area.crs
            self.area = shapely.force_2d(crop_area.geometry)
            self.area.crs = crs
            if (buffer is not None) and buffer > 0:
                self.area = buffer_in_m(self.area,buffer)


        crs = target_geoms.crs
        self.target_geoms = shapely.force_2d(target_geoms.geometry)
        self.target_geoms.crs = crs
        self.target_geoms = self.target_geoms.to_crs(4326)
        self.other_geoms = None 

        if len(self.target_geoms) == 0:
            raise Exception(f"Error loading target_paths {target_geoms}")
        
        if other_geoms is not None:
            crs = other_geoms.crs
            self.other_geoms = shapely.force_2d(other_geoms.geometry)
            self.other_geoms.crs = crs
            self.other_geoms = self.other_geoms.to_crs(4326)

            if len(self.other_geoms) == 0:
                raise Exception(f"Error loading other_paths {other_geoms}")
        
        if crop_area is not None:
            self.target_geoms = crop_gdf(self.target_geoms,self.area)

            if other_geoms is not None:
                self.other_geoms = crop_gdf(self.other_geoms, self.area)

        self.target_geoms = self.target_geoms[self.target_geoms.is_valid]

        if other_geoms is not None:
            self.other_geoms = self.other_geoms[self.other_geoms.is_valid]
            if len(self.other_geoms) == 0:
                self.other_geoms = None
                print("other_geoms is empty. Setting other_geoms to None.")


    def get(self,bounds:gpd.GeoSeries):
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)
    
        vect_anns = _linestrings_to_polygons(self.target_geoms, self.other_geoms, bounds=bounds, buffer=self.buffer,
                snap_dist_far=self.snap_dist_far, snap_dist_close=self.snap_dist_close, grid_size=self.grid_size, 
                min_target_len=self.min_target_len, max_target_len= self.max_target_len,
                min_pol_area=self.min_pol_area, max_pol_area = self.max_pol_area, 
                min_edges=self.min_edges, max_edges=self.max_edges, 
                decimals=self.decimals, prefer_small_pols=self.prefer_small_pols, 
                isolated_pols=self.isolated_pols, allow_overlap=self.allow_overlap)
        
        vect_anns = vect_anns[vect_anns.is_valid].reset_index(drop=True)
        semantic_class = list(np.ones(len(vect_anns)) * self.semantic_class)
        return gpd.GeoDataFrame({'semantic_class':semantic_class},geometry=vect_anns.geometry,crs=vect_anns.crs)
    
    def geometry(self,bounds:gpd.GeoSeries=None):
        geoms = gpd.GeoDataFrame({'semantic_class':[self.semantic_class] * len(self.target_geoms)},geometry=self.target_geoms,crs=4326)
        if bounds is not None:
                bounds = bounds.to_crs(4326)
                if len(bounds) > 1: 
                    bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

                geoms.geometry = geoms.geometry.intersection(bounds.union_all())
                geoms = geoms[geoms.geometry.is_empty == False]
        
        if self.other_geoms is not None:
            other = gpd.GeoDataFrame({'semantic_class':[0] * len(self.other_geoms)},geometry=self.other_geoms,crs=4326)
            if bounds is not None:
                other.geometry = other.geometry.intersection(bounds.union_all())
                other = other[other.geometry.is_empty == False]

            geoms = pd.concat([geoms,other.to_crs(geoms.crs)])

        return geoms[geoms.geometry.is_empty == False]
    
    def save_metadata(self,path):
        path = path.split(".")
        file_t = os.path.normpath(path[0]+"_target_geometry.geojson")
        self.target_geoms.to_file(file_t)
        print(f"target geometry saved as {file_t}")
        if self.other_geoms is not None:
            file_o = os.path.normpath(path[0]+"_other_geometry.geojson")
            self.other_geoms.to_file(file_o)
            print(f"target geometry saved as {file_o}")

        return None

def _check_isolated_pol(pols,i,prefer_small_pols:bool,min_intersection_length):
    inter = pols[i].buffer(0.01).intersection(pols)
    inter = inter[inter.is_empty == False]
    near_geoms = inter[(inter.type == "Polygon") + (inter.type == "MultiPolygon") + (inter.type == "LineString") + (inter.type == "MultiLineString")]
    near_geoms = near_geoms[near_geoms.length > min_intersection_length]
    if prefer_small_pols is True:
        add_inds = list(near_geoms.index[pols[near_geoms.index].area < pols[i].area])
    else:
        add_inds = list(near_geoms.index[pols[near_geoms.index].area > pols[i].area])

    visited_inds = []
    if len(add_inds) > 0:
        visited_inds.append(i)
        inds = []
        for j in add_inds:
            k, new_visited = _check_isolated_pol(pols,j,prefer_small_pols,min_intersection_length)
            visited_inds += new_visited
            if j in k:
                inds.append(j)
            else:
                visited_inds.append(j)

        if len(inds) == 0:
            return [i], visited_inds
        else:
            return inds, visited_inds
    else:
        return [i], visited_inds
    
def _create_isolated_pols(pols,prefer_small_pols = True,min_intersection_length=0):
    import numpy as np
    while True:
        inds = []
        visited_inds = []
        for i in pols.index:
            if i in visited_inds: 
                continue

            near_inds, new_visited_inds = _check_isolated_pol(pols,i,prefer_small_pols,min_intersection_length)
            visited_inds += new_visited_inds
            visited_inds = list(np.unique(visited_inds))
            if len(near_inds) == 1 and near_inds[0] == i:
                inds.append(i)

        inds = np.unique(inds)
        if len(inds) == len(pols):
            return pols
        else:
            pols = pols[inds].reset_index(drop=True)
        
    
def _linestrings_to_polygons(target_linestrings, other_linestrings=None, bounds=None, buffer=None,
                             snap_dist_far=0, snap_dist_close=0.1, grid_size=None, 
                             min_target_len=0, max_target_len=0, min_pol_area = 0, max_pol_area = 0, min_edges = 4, max_edges = 0,
                             decimals=3, prefer_small_pols:bool=True, isolated_pols:bool=True, allow_overlap:bool=False, has_z:bool = False):
    
    import copy
    from shapely.geometry import MultiLineString, LineString, Point
    import shapely
    

    if (bounds is not None) and (buffer is None):
        buffer = 20
    

    if grid_size is None:
        grid_size = snap_dist_close / 20 

    orig_crs = target_linestrings.crs

    if bounds is None:
        utm_crs = target_linestrings.estimate_utm_crs(datum_name='WGS 84')
    else:
        utm_crs = bounds.estimate_utm_crs(datum_name='WGS 84')

    target_linestrings = target_linestrings.to_crs(utm_crs)

    if other_linestrings is not None:
        if len(other_linestrings) == 0:
            other_linestrings = None 
        else:
            other_linestrings = other_linestrings.to_crs(utm_crs)

    if (bounds is not None) and (len(bounds) == 0 or np.all(bounds.is_empty)):
            print("bounds geometry is wrong. Setting bounds to None.")
            bounds = None

    if bounds is not None:
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

        bounds = bounds.to_crs(utm_crs)
        bounds_orig = bounds.copy()

        if buffer is not None:
            bounds = bounds.buffer(buffer,cap_style='square',join_style='mitre')

        target_linestrings = target_linestrings[target_linestrings.intersects(bounds.union_all())]
        if np.all(target_linestrings.is_empty):
            return gpd.GeoSeries([],crs=orig_crs)
        

        if other_linestrings is not None:
            other_linestrings = other_linestrings[other_linestrings.intersects(bounds.union_all())]

    target_linestrings[target_linestrings.type == "Polygon"] = target_linestrings[target_linestrings.type == "Polygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
    target_linestrings[target_linestrings.type == "MultiPolygon"] = target_linestrings[target_linestrings.type == "MultiPolygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
    target_linestrings = target_linestrings.simplify(snap_dist_close)

    if other_linestrings is None:
        other_linestrings = []
        target_linestrings = shapely.unary_union(target_linestrings)
        target_linestrings = target_linestrings.simplify(snap_dist_close)
        target_linestrings = gpd.GeoSeries(shapely.get_parts(target_linestrings),crs=utm_crs)
        target_linestrings = target_linestrings[target_linestrings.length > grid_size].reset_index(drop=True)
    else:
        other_linestrings[other_linestrings.type == "Polygon"] = other_linestrings[other_linestrings.type == "Polygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
        other_linestrings[other_linestrings.type == "MultiPolygon"] = other_linestrings[other_linestrings.type == "MultiPolygon"].buffer(0.000001,cap_style='square',join_style='mitre').boundary
        other_linestrings = other_linestrings.simplify(snap_dist_close)

        target_linestrings = gpd.GeoSeries(shapely.get_parts(target_linestrings),crs=utm_crs)
        if buffer is not None:
            other_linestrings = crop_gdf(other_linestrings,target_linestrings.buffer(buffer,cap_style='square',join_style='mitre'))

        other_linestrings = shapely.unary_union(other_linestrings)
        target_linestrings = shapely.unary_union(target_linestrings)

        inter = gpd.GeoSeries(
            shapely.get_parts(shapely.intersection(target_linestrings.buffer(snap_dist_close/2),other_linestrings)),crs=utm_crs
        )
        if inter is not None:
            inter = inter[inter.isna() == False]
            inter = inter[inter.is_empty == False]
            if len(inter) > 0:
                inter = list(inter.centroid.geometry)
                inter = shapely.geometry.MultiPoint(inter)
                target_linestrings = shapely.ops.split(target_linestrings,inter.buffer(grid_size/5))
                other_linestrings = shapely.ops.split(other_linestrings,inter.buffer(grid_size/5))   

        target_linestrings = shapely.unary_union(target_linestrings,grid_size=grid_size)
        target_linestrings = target_linestrings.simplify(snap_dist_close)
        target_linestrings = gpd.GeoSeries(shapely.get_parts(target_linestrings),crs=utm_crs)
        target_linestrings = target_linestrings[target_linestrings.length > snap_dist_close/1.25].reset_index(drop=True)

        other_linestrings = shapely.unary_union(other_linestrings,grid_size=grid_size)
        other_linestrings = other_linestrings.simplify(snap_dist_close)             
        other_linestrings = shapely.get_parts(other_linestrings)
        other_linestrings = gpd.GeoSeries(shapely.get_parts(other_linestrings),crs=utm_crs)
        other_linestrings = other_linestrings[other_linestrings.length > snap_dist_close/1.25].reset_index(drop=True)

    target_geom = shapely.unary_union(target_linestrings,grid_size=grid_size)
    union = shapely.unary_union([*target_linestrings,*other_linestrings])

    if "Multi" not in str(type(union)):
        union = MultiLineString(shapely.get_parts(union))
    
    union = union.simplify(snap_dist_close)
    union = shapely.unary_union(union,grid_size=grid_size) 

    union = [o for o in union.geoms]

    for i in range(len(union)):
        helper = MultiLineString([*union[0:i],*union[i+1:]])
        union[i] = shapely.snap(union[i],helper,snap_dist_close)
        p0 = Point(union[i].coords[0])
        p1 = Point(union[i].coords[-1])
        if shapely.intersects(p0.buffer(grid_size/2),helper) == False:
            snap_p0 = shapely.snap(p0,helper,snap_dist_far) 
        else:
            snap_p0 = p0

        if shapely.intersects(p1.buffer(grid_size/2),helper) == False:
            snap_p1 = shapely.snap(p1,helper,snap_dist_far) 
        else:
            snap_p1 = p1

        union[i] = LineString(list(snap_p0.coords) + list(union[i].coords) + list(snap_p1.coords))
        union[i] =  union[i].simplify(snap_dist_close)

    union = shapely.unary_union(union)
    pols = gpd.GeoSeries(shapely.get_parts(shapely.ops.polygonize(union)),crs=utm_crs)
    pols = pols.buffer(-grid_size,cap_style='square',join_style='mitre')
    pols = pols[pols.is_valid].reset_index(drop=True)
    pols = pols.buffer(grid_size,cap_style='square',join_style='mitre')
    pols = pols[pols.is_valid].reset_index(drop=True)
    pols = pols.simplify(snap_dist_close)

    pols = pols[pols.is_empty == False].reset_index(drop=True) 

    if max_pol_area > 0: 
        pols = pols[pols.area <= max_pol_area].reset_index(drop=True)

    if min_pol_area > 0: 
        pols = pols[pols.area >= min_pol_area].reset_index(drop=True)

    if min_edges > 0:
        pols = pols[pols.get_coordinates().index.value_counts().sort_index() >= min_edges].reset_index(drop=True)

    if max_edges > 0:
        pols = pols[pols.get_coordinates().index.value_counts().sort_index() <= max_edges].reset_index(drop=True)

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)

    if not allow_overlap:
        pols = pols[pols.buffer(-snap_dist_close/4,cap_style='square',join_style='mitre').intersects(pols.boundary.union_all()) == False].reset_index(drop=True)
        if len(pols) == 0:
            return gpd.GeoSeries([],crs=orig_crs)

    if min_target_len > 0:
        pols = pols[pols.boundary.length >= min_target_len * 2.1].reset_index(drop=True) #################################################

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)

    if (other_linestrings is not None) or min_target_len > 0 or max_target_len > 0:
        inter = pols.boundary.buffer(snap_dist_close/5,cap_style='square',join_style='mitre').intersection(target_geom)
        pols = pols[inter.isna() + inter.is_empty == 0].reset_index(drop=True)

        if min_target_len > 0 or max_target_len > 0:
            inter = inter[inter.isna() + inter.is_empty == 0].reset_index(drop=True)
            if min_target_len > 0:
                pols = pols[inter.length >= min_target_len].reset_index(drop=True)
            
            if max_target_len > 0:
                if min_target_len > 0:
                    inter = inter[inter.length >= min_target_len].reset_index(drop=True)

                pols = pols[inter.length <= max_target_len].reset_index(drop=True)

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)


    pols = pols[pols.is_valid].reset_index(drop=True) 

    if isolated_pols is True:
        pols = _create_isolated_pols(pols, prefer_small_pols=prefer_small_pols, min_intersection_length=min_target_len/4)

    if len(pols) == 0:
        return gpd.GeoSeries([],crs=orig_crs)
    else:
        pols = gpd.GeoSeries(pols,crs=utm_crs)

        if pols is None:
            return gpd.GeoSeries([],crs=orig_crs)
        else:
            pols = pols.buffer(-snap_dist_close,cap_style='square',join_style='mitre')
            pols = pols[pols.is_valid].reset_index(drop=True)
            pols = pols.buffer(snap_dist_close,cap_style='square',join_style='mitre')
            pols = pols[pols.is_valid].reset_index(drop=True)
            pols = shapely.unary_union(pols)
            pols = gpd.GeoSeries(shapely.get_parts(pols),crs=utm_crs)
            if pols is None:
                return gpd.GeoSeries([],crs=orig_crs)
            
            pols = pols[pols.is_valid].reset_index(drop=True)
            
            if bounds is None:
                return pols.to_crs(orig_crs)
            else:
                pols = pols.intersection(bounds_orig.union_all())
                if pols is None:
                    return gpd.GeoSeries([],crs=orig_crs)
                else:
                    pols = pols.buffer(-snap_dist_close,cap_style='square',join_style='mitre')
                    pols = pols[pols.is_valid].reset_index(drop=True)
                    pols = pols.buffer(snap_dist_close,cap_style='square',join_style='mitre')
                    pols = pols[pols.is_valid].reset_index(drop=True)
                    pols = shapely.unary_union(pols)
                    pols = gpd.GeoSeries(shapely.get_parts(pols),crs=utm_crs)
                    if pols is None:
                        return gpd.GeoSeries([],crs=orig_crs)

                    pols = pols[pols.is_valid].reset_index(drop=True)
                    return pols.to_crs(orig_crs)

