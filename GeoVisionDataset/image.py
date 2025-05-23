from . import raster 
import warnings
import geopandas as gpd
import numpy as np

class ImageData:
    def __init__(self,img,driver:str='JPEG') -> None:
        self.driver = driver
        self.img_obj = img
        self.crs = self.img_obj.crs

        if 'get' not in dir(img): 
            raise Exception(f".get method not implemented in image object",UserWarning)
        
        if 'basemap' not in dir(img): 
            warnings.warn(f".basemap method not implemented in image object",UserWarning)

        if 'save_metadata' not in dir(img): 
            warnings.warn(f".save_metadata method not implemented in image object",UserWarning)

    def img(self,bounds:gpd.GeoSeries,shape,dataset_bounds:gpd.GeoSeries|gpd.GeoDataFrame=None):
        from PIL import Image 
        
        bounds = bounds.copy()
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)

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
        
        img,img_bounds = self.img_obj.get(bounds,shape)

        if dataset_bounds is not None:
            intersection = img_bounds.intersection(dataset_bounds.to_crs(img_bounds.crs).union_all())
            if intersection.union_all().area < img_bounds.union_all().area:
                inside_bounds,_ = raster.rasterize(
                    intersection.geometry,
                    shape=shape,
                    bounds=img_bounds,
                    all_touched=False,
                    values=1,
                    background_index=0
                )
                img = np.array(img)
                if len(img.shape) == 3:
                    img = img * inside_bounds[:,:,np.newaxis]
                    img = Image.fromarray(img)
                else:
                    img = img * inside_bounds
                    img = Image.fromarray(img)
                    

        return img, img_bounds

    def basemap(self,bounds:gpd.GeoDataFrame|gpd.GeoSeries):
        if 'basemap' in dir(self.img_obj):
            return self.img_obj.basemap(bounds=bounds)
        else:
            return None

    def save_metadata(self,path):
        import os
        if 'save_metadata' in dir(self.img): 
            self.img.save_metadata(os.path.join(path,f"img_metadata"))
        

        


class from_files:
    def __init__(self,img_files) -> None:
        if type(img_files) == str:
            img_files = [img_files]
            
        self.crs= raster.get_crs(img_files[0])

        img_bounds = []
        for i in img_files:
            crs = raster.get_crs(i)
            if crs != self.crs:
                warnings.warn(f"Crs {crs} of img {i} is different to crs {self.crs} of first image. All img should have same crs to avoid problems.",UserWarning)

            img_bounds.append(raster.bounds(i,crs=self.crs).union_all())
        
        self.img_files = img_files
        self.img_bounds = gpd.GeoDataFrame({'filename':img_files},geometry=img_bounds,crs=self.crs)

    def get(self,bounds:gpd.GeoSeries,shape:tuple):    
        if bounds.crs != self.crs:
            warnings.warn("Changed crs of bounds. Img will cover a larger area that bounds you provided.",UserWarning)
            bounds = bounds.to_crs(self.crs)
            bounds = gpd.GeoSeries(shapely.geometry.box(*bounds.total_bounds),crs=bounds.crs)

        img_files = list(self.img_bounds['filename'][self.img_bounds.intersects(bounds.union_all())])
        
        if len(img_files) == 0:
            warnings.warn(f"No images found for bounds {str(bounds.union_all())}",UserWarning)
            return None, None

        img, img_bounds,transform = raster.merge(img_files,bounds=bounds)

        width = abs(img_bounds.total_bounds[2] - img_bounds.total_bounds[0])
        height = abs(img_bounds.total_bounds[3] - img_bounds.total_bounds[1])
        
        # Compute bounds in map units
        pixel_width = transform.a
        pixel_height = transform.e

        extent_x = abs(pixel_width * width)
        extent_y = abs(pixel_height * height)

        # If the pixel layout (width/height) doesn't match the map extent
        rotated = (width > height) != (extent_x > extent_y)

        if rotated: 
            from PIL import Image
            if len(arr.shape) == 2:
                img = Image.fromarray(np.transpose(img), mode='L')
            elif arr.shape[0] == 1:
                img = Image.fromarray(np.transpose(img[0]), mode='L')
            elif arr.shape[0] == 4:
                img = Image.fromarray(np.transpose(img, (2, 1, 0)), mode='RGBA')
            else:
                img = Image.fromarray(np.transpose(img, (2, 1, 0)), mode='RGB')
        
        else:
            img = raster.rio_to_pil(img)
            
        img = img.resize(shape)
        return img, img_bounds
    
    def basemap(self, bounds: gpd.GeoSeries = None):
        from folium.raster_layers import ImageOverlay
        from io import BytesIO
        import base64
        import warnings
    
        def pil_to_data_url(img):
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    
        if bounds is None:
            img_files = self.img_files
            img, img_bounds, _ = raster.merge(img_files)
        else:
            if len(bounds) > 1:
                bounds = gpd.GeoSeries([bounds.unary_union], crs=bounds.crs)
    
            if bounds.crs != self.crs:
                bounds = bounds.to_crs(self.crs)
    
            intersecting = self.img_bounds[self.img_bounds.intersects(bounds[0])]
            if intersecting.empty:
                warnings.warn(f"No images found for bounds {str(bounds[0])}", UserWarning)
                return None, None
    
            img_files = list(intersecting['filename'])
            img, img_bounds, _ = raster.merge(img_files, bounds=bounds)
    
        img = raster.rio_to_pil(img)
        image_url = pil_to_data_url(img)

        minx, miny, maxx, maxy = img_bounds.total_bounds
        img_bounds_list = [[miny, minx], [maxy, maxx]]

        img_overlay = ImageOverlay(
            name="Image",
            image=image_url,
            bounds=img_bounds_list,  # Make sure this is [[south, west], [north, east]]
            opacity=0.6
        )
        return img_overlay

    
    def save_metadata(self,path):
        None

def OGC(url,dataset_bounds,layer:str=None,version:str=None, output_format:str="image/jpeg",style:str=None,tilematrixset:str = None,crs=4326):
    service_type = wms_lib.get_service_type(url)
    if service_type == 'wms':
        return WMS(wms=url,layer=layer,version=version,wms_format=output_format,style=style,crs=crs)
    elif service_type == 'wmts':
        return WMTS(wmts=url,layer=layer,version=version,wmts_format=output_format,tilematrixset=tilematrixset)
    elif service_type == 'xyz':
        return XYZ(url,dataset_bounds)
    elif os.path.exists(url):
        return from_files(url) 
    else:
        raise Exception(f"Service type could not be infered or {url} is not a valid path")
        
    
class WMS:
    def __init__(self,wms,layer:str=None,version:str=None, wms_format:str="image/jpeg",style:str=None,crs=4326) -> None:
        from . import wms as wms_lib

        self.wms, self.version = wms_lib.build_wms(wms,version)
        self.layer = wms_lib.check_wms_layer(wms,layer)

        self.wms_format = wms_lib.check_wms_format(wms=self.wms,wms_format=wms_format,version=self.version)
        self.style = style

        self.crs = crs      
    
    def basemap(self,bounds:gpd.GeoSeries=None):
        from . import wms as wms_lib
        basemap = wms_lib.wms_folium_basemap(wms=self.wms,layer=self.layer,wms_format=self.wms_format)
        return basemap

    def get(self,bounds:gpd.GeoSeries, shape:tuple):
        from . import wms as wms_lib      
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)
        
        img, img_bounds = wms_lib.request_wms_image(self.wms,bounds,shape,self.layer,wms_format=self.wms_format,style=self.style,crs=self.crs)        
        return img, img_bounds
    
    def save_metadata(self,path):
        import os
        from . import wms as wms_lib
        path = path.split(".")
        path = f"{path[0]}_wms_getCapabilities.xml"
        path = os.path.normpath(path)
        wms_lib.save_wms_getcapabilities(self.wms,path)
        print(f"wms capabilities file saved as {path}")

class WMTS:
    def __init__(self,wmts,layer:str=None,version:str=None, wmts_format:str="image/jpeg",style:str=None, tilematrixset:str = None) -> None:
        from . import wms
        self.wmts, self.version = wms.build_wmts(wmts,version)
        self.layer = wms.check_wmts_layer(wmts,layer)

        self.wmts_format = wms.check_wmts_format(wmts=self.wmts,layer=self.layer,wmts_format=wmts_format,version=self.version)
        self.style = wms.check_wmts_style(wmts=self.wmts,layer=self.layer,style=style,version=self.version)
        self.tilematrixset = wms.check_wmts_tilematrixset(wmts=self.wmts,layer=self.layer,tilematrixset=tilematrixset,version=self.version)
        self.max_zoom, self.max_resolution = wms.get_wtms_max_zoom(self.wmts,layer=self.layer,wmts_format=self.wmts_format,style=self.style,
                                            tilematrixset=self.tilematrixset,version=self.version)
        
        self.crs = 4326
    
    def basemap(self,bounds:gpd.GeoSeries=None):
        from . import wms
        basemap = wms.wmts_folium_basemap(wmts=self.wmts,layer=self.layer,
                                          wmts_format=self.wmts_format,style=self.style,
                                          tilematrixset=self.tilematrixset,version=self.version)
        return basemap
    
    def get(self,bounds:gpd.GeoSeries, shape:tuple):
        from . import wms
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)
        
        img, img_bounds = wms.request_wmts_image(self.wmts,bounds,shape,layer=self.layer,wmts_format=self.wmts_format,
                                             style=self.style,tilematrixset=self.tilematrixset,max_zoom=self.max_zoom)        
        return img, img_bounds
    
    def save_metadata(self,path):
        import os
        from . import wms
        path = path.split(".")
        path = f"{path[0]}_wmts_getCapabilities.xml"
        path = os.path.normpath(path)
        wms.save_wmts_getcapabilities(self.wmts,path)
        print(f"wmts capabilities file saved as {path}")




class XYZ:
    def __init__(self,url,dataset_bounds) -> None:
        from . import wms
        if "http" not in url:
            xyz_service_urls = wms.xyz_service_urls
            url = xyz_service_urls[url]

        self.url = url
        self.dataset_bounds = dataset_bounds

        self.max_zoom, self.max_resolution = wms.get_xyz_max_zoom(url,dataset_bounds)
        
        self.crs = 4326
    
    def basemap(self,bounds:gpd.GeoSeries=None):
        from . import wms
        basemap = wms.xyz_folium_basemap(self.url)
        return basemap
    
    def get(self,bounds:gpd.GeoSeries, shape:tuple):
        from . import wms
        if len(bounds) > 1: 
            bounds = gpd.GeoSeries(bounds.union_all(),crs=bounds.crs)
        
        img, img_bounds = wms.request_xyz_image(self.url,bounds,shape,max_zoom=self.max_zoom)        
        return img, img_bounds
    
    def save_metadata(self,path):
        import os
        from . import wms
        path = path.split(".")
        path = f"{path[0]}_xyz_service_url.txt"
        path = os.path.normpath(path)
        with open(path, 'w') as file:
            file.write(self.url)

        print(f"xyz url saved as {path}")
