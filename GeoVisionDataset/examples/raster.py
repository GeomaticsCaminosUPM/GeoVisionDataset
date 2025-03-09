import geopandas as gpd 
import rasterio as rio 
import numpy as np
from pyproj import CRS
import warnings
import shapely

def rio_to_pil(arr):
    from PIL import Image
    if len(arr.shape) == 2:
        pil_image = Image.fromarray(np.transpose(arr), mode='L')
    elif arr.shape[0] == 1:
        pil_image = Image.fromarray(np.transpose(arr[0]), mode='L')
    elif arr.shape[0] == 4:
        pil_image = Image.fromarray(np.transpose(arr, (2, 1, 0)), mode='RGBA')
    else:
        pil_image = Image.fromarray(np.transpose(arr, (2, 1, 0)), mode='RGB')

    return pil_image

def pil_to_rio(img):
    from PIL import Image
    arr = np.array(img)
    if type(img) == Image:
        if len(arr.shape) > 2:
            arr = np.moveaxis(arr, -1, 0)

    return arr

def read(path,
    indexes=None, out=None, window=None, masked=False, out_shape=None, 
    resampling=rio.enums.Resampling.nearest, fill_value=None, out_dtype=None, **kwargs):

    if type(path) == str:
        src = rio.open(path)
    else:
        src = path

    
    image_array = src.read(indexes=indexes, out=out, window=window, masked=masked, out_shape=out_shape, 
                resampling=resampling, fill_value=fill_value, out_dtype=out_dtype, **kwargs)

    if src.count == 1:
        # Check if it has a color palette
        if src.colorinterp[0] == rio.enums.ColorInterp.palette:
            from cv2 import applyColorMap
            # Retrieve the color palette
            color_dict = src.colormap(1)
            image_array = colormap_to_rgb(image_array,color_dict)
            profile = src.profile
            profile.update({
                'count': 3,         # Set to 3 bands for RGB
                'dtype': 'uint8',   # Set dtype to match RGB data
            })

    elif src.count == 4:
        image_array = image_array[0:3,:,:]
        profile = src.profile
        profile.update({
            'count': 3,         # Set to 3 bands for RGB
            'dtype': 'uint8',   # Set dtype to match RGB data
        })

    metadata = src.meta

    src.close()
    return image_array, metadata

def colormap_to_rgb(img,colormap):
    from cv2 import applyColorMap
    colormap_matrix = np.zeros((256, 1, 3), dtype=np.uint8)
    for key, color in colormap.items():
        colormap_matrix[key] = color[0:3]  # Assign color for each key
                
    img = np.transpose(applyColorMap(img[0,:,:], colormap_matrix),(2,1,0)).astype(np.uint8)
    return img 

def validate_crs(src:rio.io.DatasetReader|rio.io.DatasetWriter|dict|CRS|str|int):
    import re
    if type(src) is dict:
        crs = src['crs']
    elif type(src) is CRS:
        crs = src
    elif type(src) is str:
        crs = CRS.from_string(src)
    elif type(src) is int:
        crs = CRS.from_epsg(src)
    else:
        try:
            crs = src.crs 
        except:
            raise Exception(f"src type {type(src)} not accepted: {src}")

    warnings.filterwarnings(
        "ignore",
        message=re.escape(
            "You will likely lose important projection information when converting to a PROJ string from another format. See: https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems"
        ),
        category=UserWarning
    )
    if len(crs.to_proj4()) == 0:
        crs_str = crs.to_wkt()
        
        if "LOCAL_CS" in crs_str:
            if "ETRS89-extended / LAEA Europe" in crs_str:
                crs = CRS.from_epsg(3035)

                if (type(src) == rio.io.DatasetReader) and (src.mode != 'r'):
                    src.crs = crs

                return crs
            # Add more mappings as needed
            # elif "Another projection" in crs_str:
            #     return CRS.from_epsg(some_epsg_code)
            else:
                raise ValueError("Unknown LOCAL_CS definition; manual intervention needed.")
        else:
            raise ValueError("CRS is invalid, but not due to LOCAL_CS.")
    else:
        return crs.to_epsg() # to_proj4()

def driver_and_extension(driver):
    if driver == 'GTiff' or driver == "tif" or driver == ".tif":
        extension = ".tif"
        driver = 'GTiff'
    elif driver == 'JPEG' or driver == "jpg" or driver == ".jpg": 
        extension = ".jpg"
        driver = 'JPEG'
    elif driver == 'PNG' or driver == "png" or driver == ".png": 
        extension = ".png"
        driver = 'PNG'
    else:
        raise Exception(f"driver {driver} not implemented")
    
    return driver, extension

def get_crs(file):
    if type(file) is str:
        try:
            src = rio.open(file,'r+')
        except:
            src = rio.open(file,'r')
    else:
        src = file

    crs = validate_crs(src)
    if type(file) is str:    
        src.close()
    
    return crs

def bounds(file,crs=None):
    from pyproj import Transformer
    from pyproj import CRS
    
    if type(file) is str:
        import rasterio as rio
        try:
            src = rio.open(file,'r+')
        except:
            src = rio.open(file,'r')

        file_crs = CRS.from_epsg(get_crs(src))
        src.close()
    else:
        src = file
        file_crs = get_crs(src)

    r = gpd.GeoSeries(shapely.geometry.box(src.bounds.left,src.bounds.bottom,src.bounds.right,src.bounds.top),crs=file_crs)
    if crs is not None: 
        r = r.to_crs(crs)

    return r

def raster_bounds(file,crs=None):
    return bounds(file,crs=crs)

def bites_to_image(s):
    from PIL import Image
    from io import BytesIO
    image_data = BytesIO(s)
    image = Image.open(image_data)  
    return image

def merge(input_paths,bounds:gpd.GeoSeries=None,get_bounds:bool=False,get_transform:bool=False):
    from rasterio.merge import merge
    #from rasterio.features import geometry_mask
    import numpy as np
    from PIL import Image

    # Open all input GeoTIFF files
    if type(input_paths[0]) == str:
        src = [rio.open(path) for path in input_paths]
    else:
        src = input_paths

    crs = validate_crs(src[0])

    if bounds is None:
        mosaic, out_trans = merge(src)
    else:    
        orig_crs = bounds.crs 
        bounds = bounds.to_crs(crs)
        mosaic, out_trans = merge(src,bounds=tuple(bounds.total_bounds)) 
        #mask = geometry_mask(bounds,out_shape=mosaic.shape[1:],invert=False,transform=out_trans)
        #mosaic = np.array(np.ma.masked_array(mosaic, mask))
    
    if (src[0].count == 1) and (src[0].colorinterp[0] == rio.enums.ColorInterp.palette):
        mosaic = colormap_to_rgb(mosaic,src[0].colormap(1))

    for i in src:
        i.close()

    # Get raster dimensions
    _,width,height = mosaic.shape

    # Calculate bounds using array_bounds
    img_bounds = rio.transform.array_bounds(height, width, out_trans)
    img_bounds = gpd.GeoSeries(shapely.geometry.box(*img_bounds),crs=crs)
    #if bounds is not None:
    #    img_bounds = img_bounds.to_crs(orig_crs)

    return mosaic, img_bounds, out_trans  

def crop(input_path, bounds:gpd.GeoSeries):
    from rasterio.windows import from_bounds#, bounds as window_bounds
    # Open the GeoTIFF file
    if type(input_path) == str:
        try:
            src = rio.open(input_path,'r+')
        except:
            src = rio.open(input_path,'r')
    else:
        src = input_path

    image_bounds = raster_bounds(src)
    bounds = bounds.to_crs(image_bounds.crs)
    bounds = bounds.intersection(image_bounds)
    new_image_bounds = gpd.GeoSeries(shapely.geometry.box(*bounds.total_bounds),crs=image_bounds.crs)
    # Get the window coordinates based on the bounding box
    window = from_bounds(*bounds.total_bounds, transform=src.transform)
    # Read the data from the specified window
    cropped_data,meta = read(src,window=window)

    meta['width'], meta['height'] = window.width, window.height
    meta['transform'] = src.window_transform(window)
    meta['crs'] = image_bounds.crs
    src.close()
    return cropped_data, new_image_bounds, meta

def pixels_to_gdf(raster, meta, bounds:gpd.GeoSeries=None):
    from shapely import prepare, intersects

    # Extract metadata
    transform = meta['transform']
    height, width = raster.shape[1:]

    # Initialize lists to store geometries and pixel values
    geometries = []
    values = []

    # Iterate over each pixel in the window
    for i in range(height):
        for j in range(width):
            # Get pixel value
            value = raster[0, i, j]  # Accessing the first band assuming single-band raster

            # Calculate pixel coordinates in CRS
            lon, lat = rio.transform.xy(transform, i, j)

            # Create bounding box geometry for the pixel
            minx, miny = rio.transform.xy(transform, i - 0.5, j - 0.5)
            maxx, maxy = rio.transform.xy(transform, i + 0.5, j + 0.5)
            pixel_box = shapely.geometry.box(minx, miny, maxx, maxy)

            geometries.append(pixel_box)
            values.append(value)

    # Create a GeoDataFrame
    df = gpd.GeoDataFrame({'value': values, 'geometry': geometries}, crs=validate_crs(meta['crs']))

    if bounds is not None:
        bounds = bounds.to_crs(df.crs)
        geoms = list(df.geometry.centroid)
        prepare(geoms)
        df = df.loc[intersects(geoms,bounds.geometry.union_all())].reset_index(drop=True)


    return df

def vectorize(imgae, bounds, background_value=0, simplify = 0, buffer=0, min_area=0):
    crs = bounds.crs
    image = pil_to_rio(image)
    mask = image != background_value  # Create a mask to exclude the background

    width, height = image.shape
    transform = rio.transform.from_bounds(*bounds.total_bounds, width, height)
    
    # Vectorize the raster
    results = (
        {'properties': {'pixel_values': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            rio.features.shapes(image, mask=mask, transform=transform)
        )
    )
    
    # Convert to GeoDataFrame
    geometries = list(results)
    gdf = gpd.GeoDataFrame.from_features(geometries)
    
    # Set the coordinate reference system (CRS) of the GeoDataFrame
    gdf.set_crs(crs, inplace=True)
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    labels = []
    geoms = []
    for i in np.unique(gdf['pixel_values']):
        g = gdf.geometry[gdf['pixel_values'] == i]
        if (simplify > 0) or (buffer > 0) or (min_area) > 0:
            g = g.simplify(simplify)
            g = g.buffer(-buffer*0.5)
            g = g[g.area > min_area]
            g = shapely.unary_union(g.buffer(buffer))
            g = gpd.GeoSeries(shapely.get_parts(g),crs=gdf.crs)
            g = g.buffer(-buffer*0.5)
            
        for j in g:
            geoms.append(j)
            labels.append(i)

    gdf = gpd.GeoDataFrame({"label":labels},geometry=gpd.GeoSeries(geoms,crs=gdf.crs),crs=gdf.crs)    

    return gdf.to_crs(crs)

def rasterize(gdf:gpd.GeoDataFrame|gpd.GeoSeries,shape:tuple|int,bounds:gpd.GeoDataFrame|gpd.GeoSeries=None,all_touched:bool=False,values=None,background_index:int=0):
    from rasterio.transform import from_bounds
    from rasterio import features

    if values is None:
        try:
            values = list(gdf['values'].astype(int))
        except:
            values = 1 
            
    elif type(values) == str:
        values = list(gdf[values].astype(int))

    if type(values) == int:
        values = [values]

    if len(gdf) != len(values):
        if len(values) == 1:
            for _ in range(len(gdf) - len(values)):
                values.append(values[-1])
        else:
            raise Exception(f"The values list has a length of {len(values)} but we have {len(gdf)} number of geometries.")

    if background_index is None:
        if type(values) == list:
            background_index = int(max(values)+1)

    gdf = gpd.GeoDataFrame({'values':values},geometry=gdf.geometry,crs=gdf.crs)
    gdf = gdf[gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.is_empty == False]
    gdf = gdf.reset_index(drop=True)
    values = list(gdf['values'].astype(int))

    if len(gdf) == 0:
        return np.ones(shape,dtype=uint8) * background_index
    
    if type(values) != list:
        raise Exception(f"Values should be a list but got {type(values)} values = {values}")

    if bounds is None:
        minx,miny,maxx,maxy = gdf.total_bounds
        img_bounds = gpd.GeoSeries(shapely.geometry.box(minx,miny,maxx,maxy),crs=bounds.crs)
    else:
        utm = bounds.estimate_utm_crs()
        gdf = gdf.to_crs(bounds.crs)
        gdf = gdf.intersection(bounds.union_all())
        minx,miny,maxx,maxy = bounds.total_bounds 
        img_bounds = gpd.GeoSeries(shapely.geometry.box(minx,miny,maxx,maxy),crs=bounds.crs)
        if abs(bounds.to_crs(utm).union_all().area - img_bounds.to_crs(utm).union_all().area) > 10**-5:
            warnings.warn(
                "The rasterized image bounds do not match exactly with the provided bounds. It covers a larger area.",
                UserWarning
            )

        if type(shape) is int:
            dx = (maxx - minx)
            dy = (maxy - miny)
            if dx < dy:
                shape = (shape, int(dy/dx * shape))
            else:
                shape = (int(dx/dy * shape), shape)
            
    if type(shape) is tuple:
        if len(shape) != 2:
            raise Exception(f"shape is wrong {shape}")
    elif type(shape) is int:
        raise Exception(f"If shape is int {shape} you should set the bounds keyword to a geoseries object.")

    shape = (int(shape[0]),int(shape[1]))
    #resx = (maxx - minx) / shape[0]
    #resy = (maxy - miny) / shape[1]
    transform = from_bounds(minx, miny, maxx, maxy, shape[0], shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = np.array(values)[gdf.geometry.area > 0]
        gdf = gdf.geometry[gdf.geometry.area > 0].reset_index(drop=True)

    geometries = ((gdf.geometry[i],values[i]) for i in gdf.index)
    
    raster = features.rasterize(geometries,
                                out_shape = (shape[1],shape[0]),
                                fill = background_index,
                                out = None,
                                transform=transform,
                                all_touched = all_touched,
                                default_value = background_index,
                                dtype = np.min_scalar_type(np.max(values)))
    return raster, img_bounds

def save(output_path:str, arr, bounds:gpd.GeoSeries, driver:str = "JPEG"):
    # driver available here https://gdal.org/drivers/raster/index.html
    driver, extension = driver_and_extension(driver)
    output_path = output_path.split('.')[0] + extension

    arr = pil_to_rio(arr)

    # Extract the GeoSeries bounds
    crs = bounds.crs
    img_bounds = gpd.GeoSeries(shapely.geometry.box(*bounds.total_bounds),crs=crs)
    utm = img_bounds.estimate_utm_crs()
    if abs(bounds.to_crs(utm).union_all().area - img_bounds.to_crs(utm).union_all().area) > 10**-4:
        warnings.warn(
            "The rasterized image bounds do not match exactly with the provided bounds. It covers a larger area.",
            UserWarning
        )
        
    minx,miny,maxx,maxy = img_bounds.total_bounds

    if len(arr.shape) == 2:
        arr = arr[np.newaxis,:,:]
        
    if len(arr.shape) == 3:
        n_bands = arr.shape[0]
        y_shape = arr.shape[1]
        x_shape = arr.shape[2]
    else:
        raise Exception(f"shape of array is {arr.shape} but is should have the shape (n_bands,x_shape,y_shape)")

    if n_bands > 3: 
        raise Exception("The array should have the shape (n_bands,x_shape,y_shape)")
        
    # Calculate the pixel size
    x_pixel_size = (maxx - minx) / x_shape
    y_pixel_size = (maxy - miny) / y_shape

    # Create the transformation
    transform = rio.transform.from_bounds(minx, miny, maxx, maxy, x_shape, y_shape)

    
    dtype = np.min_scalar_type(np.max(arr))
    if 'int64' in str(dtype):
        dtype = np.float64

    if 'uint16' in str(dtype):
        dtype = np.uint32

    if 'int16' in str(dtype):
        dtype = np.int32

    if 'float16' in str(dtype):
        dtype = np.float32
        
    # Create the GeoTIFF file
    with rio.open(
        output_path,
        'w',
        driver=driver,
        height=y_shape,
        width=x_shape,
        count=n_bands,  # Number of bands (RGB)
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr)  



