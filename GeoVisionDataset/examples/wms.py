import geopandas as gpd
import shapely
import pyproj
import warnings 
# wms functions 

xyz_service_urls = {
    'Google Maps': 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
    'Google Satellite': 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    'Google Terrain': 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
    'Google Satellite Hybrid': 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    'Esri Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'OpenStreetMap': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
}

def buffer_in_m(geom,buffer):
    orig_crs = geom.crs
    if orig_crs.is_projected == False:
        utm = geom.to_crs(geom.estimate_utm_crs(datum_name='WGS 84'))
    
    utm = utm.buffer(buffer)
    return utm.to_crs(orig_crs)


def bites_to_image(s):
    from PIL import Image
    from io import BytesIO
    image_data = BytesIO(s)
    image = Image.open(image_data)  
    return image

def extract_wms_info_from_xml(xml_content):
    import xml.etree.ElementTree as ET
    try:
        # Define the namespace dictionary
        namespaces = {
            'wms': 'http://www.opengis.net/wms',
            'xlink': 'http://www.w3.org/1999/xlink'
        }

        # Parse the XML content using ElementTree
        root = ET.fromstring(xml_content)

        # Extract WMS version from the root element
        wms_version = root.get('version')

        # Find the WMS service URL using namespaces
        wms_url_element = root.find('.//wms:GetCapabilities/wms:DCPType/wms:HTTP/wms:Get/wms:OnlineResource[@xlink:type="simple"]', namespaces=namespaces)
        wms_url = wms_url_element.get('{http://www.w3.org/1999/xlink}href')

        return wms_version, wms_url
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def extract_wmts_info_from_xml(xml_content):
    import xml.etree.ElementTree as ET

    # Define the namespace
    namespace = {"wmts": "http://www.opengis.net/wmts/1.0", "xlink": "http://www.w3.org/1999/xlink"}

    # Parse the XML data
    root = ET.fromstring(xml_content)

    try:
        # Find the ServiceMetadataURL element
        service_metadata_element = root.find(".//wmts:ServiceMetadataURL", namespace)

        # Check if the element is present
        if service_metadata_element is not None:
            # Get the value of the xlink:href attribute
            service_metadata_url = service_metadata_element.get("{http://www.w3.org/1999/xlink}href")

    except Exception as e:
        print("Error:", str(e))
    
    if service_metadata_element is None:
        raise Exception("ServiceMetadataURL element not found in the XML.")
    
    return service_metadata_url


def wms_from_xml(xml_file):
    from owslib.wms import WebMapService
    xml_file = open(xml_file).read()
    version, url = extract_wms_info_from_xml(xml_file)
    try:
        # Create WebMapService instance directly from XML file
        wms = WebMapService(url, version= version, xml=xml_file)

        return wms
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
def wmts_from_xml(xml_file):
    from owslib.wmts import WebMapTileService

    xml_file = open(xml_file).read()
    url = extract_wmts_info_from_xml(xml_file)
    try:
        # Create WebMapService instance directly from XML file
        wms = WebMapTileService(url, xml=xml_file)

        return wms
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_wms_version(wms_url:str):
    import re
    version = re.findall("\d.\d.\d",wms_url)
    if len(version) == 0:
        raise Exception(f"version was not found in url {wms_url}")
    version = version[0]
    return version

def get_wmts_version(wmts_url:str):
    import requests
    import xml.etree.ElementTree as ET


    # Make an HTTP request to the URL
    response = requests.get(wmts_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the XML data from the response
        root = ET.fromstring(response.text)

        # Extract the version attribute from the root element
        wmts_version = root.get("version")

        if wmts_version is not None:
            return wmts_version
        else:
            raise Exception("WMTS version attribute not found in the XML.")

    else:
        raise Exception(f"Error: Unable to fetch data. Status code: {response.status_code}")

def get_wfs_version(wfs_url:str):
    import requests
    import xml.etree.ElementTree as ET


    # Make an HTTP request to the URL
    response = requests.get(wfs_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the XML data from the response
        root = ET.fromstring(response.text)

        # Extract the version attribute from the root element
        wfs_version = root.get("version")

        if wfs_version is not None:
            return wfs_version
        else:
            raise Exception("WMTS version attribute not found in the XML.")

    else:
        raise Exception(f"Error: Unable to fetch data. Status code: {response.status_code}")

def get_wms_layer_in_url(wms_url:str):
    if wms_url.find("layers=") == -1:
        raise Exception(f"layer was not found in url {wms_url}")
    layers = wms_url[wms_url.find("layers=")+len("layers="):]
    if layers.find("&") != -1:
        layers = layers[:layers.find("&")]
    return layers

################################################################################################################################ añadir

def build_wms(wms,version:str=None):
    from owslib.wms import WebMapService
    if type(wms) is str and ".xml" in wms and "http" not in wms:
        wms = wms_from_xml(wms)

    if type(wms) is str:
        if version is None:
            version = get_wms_version(wms)

        wms = WebMapService(wms,version=version)

    version = wms.identification.version
    return wms, version

def build_wmts(wmts,version:str=None):
    from owslib.wmts import WebMapTileService
    if type(wmts) is str and ".xml" in wmts and "http" not in wmts:
        wmts = wmts_from_xml(wmts)
    if type(wmts) is str:
        if version is None:
            version = get_wmts_version(wmts)
        wmts = WebMapTileService(wmts,version=version)
    version = wmts.version
    return wmts, version   

def build_wfs(wfs,version:str=None):
    from owslib.wfs import WebFeatureService
    if type(wfs) is str and ".xml" in wfs and "http" not in wfs:
        wfs = wmts_from_xml(wfs) #########################################
    if type(wfs) is str:
        if version is None:
            version = get_wfs_version(wfs)
        wfs = WebFeatureService(wfs,version=version)
    version = wfs.version
    return wfs, version   

def check_wms_layer(wms,layer:str=None,version:str=None):
    wms,version = build_wms(wms,version)
    url = wms.url
    layers = list(wms.contents)
    if len(layers) == 0:
        raise Exception("No layers found")
    if layer in layers:
        return layer 
    else:
        print(f"Layer {layer} not found. Changed to {layers[0]}")
        return layers[0]
    
def check_wmts_layer(wmts,layer:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version)
    url = wmts.url
    layers = list(wmts.contents)
    if len(layers) == 0:
        raise Exception("No layers found")
    if layer in layers:
        return layer 
    else:
        print(f"Layer {layer} not found. Changed to {layers[0]}")
        return layers[0]
    
def check_wfs_typename(wfs,typename:str=None,version:str=None):
    wfs,version = build_wfs(wfs,version)
    url = wfs.url
    typenames = list(wfs.contents)
    if len(typename) == 0:
        raise Exception("No typenames found")
    
    if typename in typenames:
        return typename 
    else:
        print(f"Typename {typename} not found. Changed to {typenames[0]}")
        return typenames[0]
    
def check_wms_layers(wms,layers=None,version:str=None,include_not_found:bool=True):
    wms, version = build_wms(wms,version)
    if type(layers) != list and type(layers) != tuple:
        layers = [layers]
    _layers = []
    for l in layers:
        if l is None:
            _l = check_wms_layer(wms,None)

        else:
            _l = check_wms_layer(wms,l)
        if l == _l or include_not_found:
            l = _l
            if l not in _layers:
                _layers.append(l)
    return _layers

def check_wmts_layers(wmts,layers=None,version:str=None,include_not_found:bool=True):
    wmts, version = build_wmts(wmts,version)
    if type(layers) != list and type(layers) != tuple:
        layers = [layers]
    _layers = []
    for l in layers:
        if l is None:
            _l = check_wmts_layer(wmts,None)
        else:
            _l = check_wmts_layer(wmts,l)
        if l == _l or include_not_found:
            l = _l
            if l not in _layers:
                _layers.append(l)
    return _layers

def check_wfs_typenames(wfs,typenames=None,version:str=None,include_not_found:bool=True):
    wfs, version = build_wfs(wfs,version)
    if type(typenames) != list and type(typenames) != tuple:
        typenames = [typenames]
    _typenames = []
    for l in typenames:
        if l is None:
            _l = check_wfs_typename(wfs,None)
        else:
            _l = check_wfs_typename(wfs,l)
        if l == _l or include_not_found:
            l = _l
            if l not in _typenames:
                _typenames.append(l)

    return _typenames

def check_wms_format(wms,wms_format:str=None, version:str=None):
    wms,version = build_wmts(wms,version)
    fo = wms.getOperationByName('GetMap').formatOptions
    if len(fo) == 0:
        raise Exception(f"No format options found")
    if wms_format in fo:
        return wms_format
    else:
        print(f"Format {wms_format} not found. Changed to {fo[0]}")
        return fo[0]    
    
def check_wmts_format(wmts,layer:str = None,wmts_format:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer)
    fo = wmts[layer].formats
    if len(fo) == 0:
        raise Exception(f"No format options found")
    if wmts_format in fo:
        return wmts_format
    else:
        print(f"Format {wmts_format} not found. Changed to {fo[0]}")
        return fo[0] 


def check_wfs_format(wfs,typename:str = None,wfs_format=None,version:str=None):
    wfs,version = build_wfs(wfs,version)
    typename = check_wfs_typename(wfs,typename=typename)
    fo = wfs[typename].outputFormats
    if len(fo) == 0:
        return None
        #fo = wfs.getOperationByName('GetFeature').formatOptions

    #if len(fo) == 0:
    #    raise Exception(f"No format options found")

    if wfs_format in fo:
        return wfs_format
    else:
        print(f"Format {wfs_format} not found. Changed to {fo[0]}")
        return fo[0] 
    

def check_wfs_crs(wfs,typename:str = None,crs=None,version:str=None):
    import pyproj
    import numpy as np
    wfs,version = build_wfs(wfs,version)
    typename = check_wfs_typename(wfs,typename=typename)
    if crs is not None:
        crs = pyproj.CRS.from_user_input(crs).to_epsg()
    else:
        crs = None

    crss = [pyproj.CRS.from_user_input(str(i)).to_epsg() for i in wfs[typename].crsOptions]
    crss_orig = np.array([str(i) for i in wfs[typename].crsOptions])

    if len(crss) == 0:
        raise Exception(f"No crs options found")
    if crs in crss:
        return str(crss_orig[np.array(crss) == crs][0])
    else:
        print(f"Crs {crs} not found. Changed to {crss[0]}")
        return str(crss_orig[0])


def check_wmts_style(wmts,layer:str=None,style:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer)
    try:
        s = list(wmts[layer].styles.keys())
    except:
        s = []

    if len(s) == 0:
        raise Exception(f"No styles options found")
    
    if style in s:
        return style
    else:
        print(f"Style {style} not found. Changed to {s[0]}")
        return s[0] 

    
def check_wfs_style(wfs,typename:str=None,style:str=None,version:str=None):
    wfs,version = build_wfs(wfs,version)
    typename = check_wfs_typename(wfs,typename=typename)
    try:
        s = list(wfs[typename].styles.keys())
    except:
        s = []

    if len(s) == 0:
        raise Exception(f"No styles options found")
    
    if style in s:
        return style
    else:
        print(f"Style {style} not found. Changed to {s[0]}")
        return s[0] 
    
def check_wmts_tilematrixset(wmts,layer:str=None,tilematrixset:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer)
    t = list(wmts[layer].tilematrixsetlinks.keys())
    if len(t) == 0:
        raise Exception(f"No tilematrixsets found")
    if tilematrixset in t:
        return tilematrixset
    else:
        print(f"Tilematrixset {tilematrixset} not found. Changed to {t[0]}")
        return t[0]

def check_wmts_zoom(wmts,zoom,layer:str=None,tilematrixset:str=None,version:str=None):
    wmts,version = build_wms(wmts,version=version)
    if type(zoom) is str:
        if zoom != "max" and zoom != "min":
            zoom = int(zoom)

    tilematrixset = check_wmts_tilematrixset(wmts,tilematrixset=tilematrixset,layer=layer,version=version)
    z = [i for i in wmts.tilematrixsets[tilematrixset].tilematrix.keys()]
    if len(z) == 0:
        raise Exception(f"No zoom levels found")
    if str(zoom) in z:
        return zoom
    else:
        if zoom == "max":
            return z[-1]
        elif zoom == "min":
            return z[0]
        
        try:
            zoom = int(zoom)
        except:
            raise Exception(f"zoom {zoom} not available. Available zoom levels are {[i for i in range(len(z))]}")
        
        if zoom > len(z):
            zoom_b = z[-1]
        elif zoom < 0:
            zoom_b = z[0]
        else:
            return z[zoom]
        
        print(f"Zoom {zoom} not available. Changed to {zoom_b}")
        return zoom_b   

def wms_layer_bbox(wms,layer:str=None,version:str=None):
    wms,version = build_wms(wms,version=version)
    layer = check_wms_layer(wms,layer,version)
    # Extract bounding box information
    minx, miny, maxx, maxy = wms[layer].boundingBoxWGS84
    return minx,miny,maxx,maxy

def wmts_layer_bbox(wmts,layer:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version=version)
    layer = check_wmts_layer(wmts,layer,version)
    # Extract bounding box information
    minx, miny, maxx, maxy = wmts[layer].boundingBoxWGS84
    return minx,miny,maxx,maxy
    
def wfs_typename_bbox(wfs,typename:str=None,version:str=None):
    wfs,version = build_wfs(wfs,version=version)
    typename = check_wfs_typename(wfs,typename,version)
    # Extract bounding box information
    minx, miny, maxx, maxy = wfs[typename].boundingBoxWGS84
    return minx,miny,maxx,maxy
    
def wms_layer_center(wms,layer:str=None,version:str=None):
    minx,miny,maxx,maxy = wms_layer_bbox(wms,layer,version)
    # Calculate center coordinates
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    return (center_x,center_y)

def wmts_layer_center(wmts,layer:str=None,version:str=None):
    minx,miny,maxx,maxy = wmts_layer_bbox(wmts,layer,version)
    # Calculate center coordinates
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    return (center_x,center_y)

def wfs_typename_center(wfs,typename:str=None,version:str=None):
    minx,miny,maxx,maxy = wfs_typename_bbox(wfs,typename,version)
    # Calculate center coordinates
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    return (center_x,center_y)

def wmts_resource_url(wmts,layer:str=None,wmts_format:str=None,style:str=None,tilematrixset:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version=version)
    layer = check_wmts_layer(wmts,layer=layer,version=version)
    wmts_format = check_wmts_format(wmts,wmts_format=wmts_format,layer=layer,version=version)
    style = check_wmts_style(wmts,style=style,layer=layer,version=version)
    tilematrixset = check_wmts_tilematrixset(wmts,tilematrixset=tilematrixset,layer=layer,version=version)
    urls_list = wmts[layer].resourceURLs
    valid_urls = []
    for u in urls_list:
        if wmts_format == u["format"]:
            if u['resourceType'] == 'tile':
                temp = u['template']
                temp=temp.replace("{Style}",style)
                temp=temp.replace("{TileMatrixSet}",tilematrixset)
                temp=temp.replace("{TileMatrix}","{z}")
                temp=temp.replace("{TileRow}","{y}")
                temp=temp.replace("{TileCol}","{x}")
                valid_urls.append(temp)
    if len(valid_urls) == 0:
        raise Exception(f"No valid resource urls found with format {wmts_format} and resourceType 'tile'")
    elif len(valid_urls) > 1:
        print("Multiple valid urls found. Returning first url")
        for i in valid_urls:
            print(i)

    return valid_urls[0]

def save_wms_getcapabilities(wms, output_file, version:str=None):
    wms, version = build_wms(wms,version)
    try:
        # Get the GetCapabilities response
        capabilities_xml = wms.getServiceXML()

        # Save the XML content to the specified output file
        with open(output_file, 'wb') as file:
            file.write(capabilities_xml)
        
        print(f"GetCapabilities XML saved to {output_file}")
    except Exception as e:
        print(f"An error occurred saving GetCapabilities XML: {str(e)}")

def save_wmts_getcapabilities(wmts, output_file, version:str=None):
    wmts, version = build_wmts(wmts,version)
    try:
        # Get the GetCapabilities response
        capabilities_xml = wmts.getServiceXML()

        # Save the XML content to the specified output file
        with open(output_file, 'wb') as file:
            file.write(capabilities_xml)
        
        print(f"GetCapabilities XML saved to {output_file}")
    except Exception as e:
        print(f"An error occurred saving GetCapabilities XML: {str(e)}")

def save_wfs_getcapabilities(wfs, output_file, version:str=None):
    wfs, version = build_wfs(wfs,version)
    try:
        # Get the GetCapabilities response
        capabilities_xml = wfs.getServiceXML()

        # Save the XML content to the specified output file
        with open(output_file, 'wb') as file:
            file.write(capabilities_xml)
        
        print(f"GetCapabilities XML saved to {output_file}")
    except Exception as e:
        print(f"An error occurred saving GetCapabilities XML: {str(e)}")


def wmts_basemap(wmts,layer:str=None,wmts_format = 'image/jpeg',style:str=None,tilematrixset:str=None,version:str=None):
    import xyzservices as xyz
    wmts, version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer)
    wmts_format = check_wmts_format(wmts=wmts,layer=layer,wmts_format=wmts_format,version=version)
    style = check_wmts_style(wmts=wmts,layer=layer,style=style,version=version)
    tilematrixset = check_wmts_tilematrixset(wmts=wmts,layer=layer,tilematrixset=tilematrixset,version=version)
    tp = xyz.TileProvider(
        name="WMTS",
        url=wmts_resource_url(wmts,layer=layer,wmts_format=wmts_format,style=style,tilematrixset=tilematrixset,version=version),
        attribution=None,
    )
    return tp

def xyz_basemap(url):
    import xyzservices as xyz
    if "http" not in url:
        url = xyz_service_urls[url]

    tp = xyz.TileProvider(
        name="XYZ",
        url=url,
        attribution=None,
    )
    return tp

def wms_leaflet_basemap(wms,layer:str=None,wms_format = 'image/jpeg',version:str=None,transparent:bool=False):
    from ipyleaflet import WMSLayer
    wms, version = build_wms(wms,version)
    url = wms.url
    name = wms.identification.title
    if type(layer) is list:
        layer = layer[0]
    layer = check_wms_layer(wms,layer)
    wms_format = check_wms_format(wms,wms_format)

    lmap = WMSLayer(
        **{'url':wms.url,
        'layers':layer,
        #'format':wms_format,
        'transparent':transparent,
        'name':name}
    )
    return lmap

def wmts_leaflet_basemap(wmts,layer:str=None,wmts_format = 'image/jpeg',style:str=None,tilematrixset:str=None,version:str=None,transparent:bool=False): 
    from ipyleaflet import WMSLayer
    wmts, version = build_wmts(wmts=wmts,version=version)
    name = wmts.identification.title
    basemap = wmts_basemap(wmts,layer=layer,wmts_format=wmts_format,style=style,tilematrixset=tilematrixset,version=version)
    lmap = WMSLayer(
        **{'url':basemap.build_url(),
        'layers':layer,
        #'format':wms_format,
        'transparent':transparent,
        'name':name}
    )
    return lmap

def xyz_leaflet_basemap(url,transparent:bool=False,name:str="XYZ"): 
    from ipyleaflet import WMSLayer
    basemap = xyz_basemap(url)
    lmap = WMSLayer(
        **{'url':basemap.build_url(),
        'layers':layer,
        #'format':wms_format,
        'transparent':transparent,
        'name':name}
    )
    return lmap

def wms_folium_basemap(wms,layer:str=None,wms_format = 'image/jpeg',version:str=None,transparent:bool=False):
    import folium
    wms, version = build_wms(wms,version)
    url = wms.url
    name = wms.identification.title
    #if type(layer) is list:
    #    layer = layer[0]
    layer = check_wms_layer(wms,layer)
    wms_format = check_wms_format(wms,wms_format)
    fmap = folium.raster_layers.WmsTileLayer(url = url,
                            layers = layer,
                            transparent = transparent, 
                            fmt=wms_format,
                            name = name
                            )
    return fmap

def wmts_folium_basemap(wmts,layer:str=None,wmts_format = 'image/jpeg',style:str=None,tilematrixset:str=None,version:str=None,transparent:bool=False):
    import folium
    wmts, version = build_wmts(wmts=wmts,version=version)
    name = wmts.identification.title
    basemap = wmts_basemap(wmts,layer=layer,wmts_format=wmts_format,style=style,tilematrixset=tilematrixset,version=version)
    fmap = folium.TileLayer(
            tiles=basemap.build_url(),
            attr='None',
            name=name
        )
    return fmap

def xyz_folium_basemap(url,transparent:bool=False,name:str="XYZ"):
    import folium
    basemap = xyz_basemap(url)
    fmap = folium.TileLayer(
            tiles=basemap.build_url(),
            attr='None',
            name=name
        )
    return fmap


def request_wfs_features(wfs,bounds:gpd.GeoSeries,typename:str=None,wfs_format:str=None,version:str=None,crs=None):
    import pyproj

    if crs is None:
        try:
            crs = bounds.crs
        except:
            crs = 4326 

    if type(crs) is not str:
        crs = str(crs)

    orig_crs = bounds.crs

    wfs, version = build_wfs(wfs,version)


    typename = check_wfs_typename(wfs,typename=typename,version=version)


    wfs_format = check_wfs_format(wfs,wfs_format=wfs_format,typename=typename, version=version)
    crs = check_wfs_crs(wfs,typename=typename,crs=crs,version=version)
    bounds = bounds.to_crs(pyproj.CRS.from_user_input(crs))

    features = wfs.getfeature(typename=typename,
                        srsname=crs,
                        bbox=tuple(bounds.total_bounds),
                        outputFormat=wfs_format
                        )
        

    gdf = gpd.read_file(bytes(features.read()))
    gdf.crs = pyproj.CRS.from_user_input(crs) 
    gdf = gdf.to_crs(orig_crs)
    #gdf = utils.intersect_geoseries(gdf,bounds)

    return gdf
        
    
def request_wms_image(wms,bounds:gpd.GeoSeries,shape:tuple,layer:str=None,wms_format:str=None,version:str=None,style=None,transparent:bool=False,crs=None):
    if type(style) != list and type(style) != tuple:
        style = [style]

    if crs is None:
        crs = bounds.crs

    if type(crs) is not str:
        crs = str(crs)

    crs = pyproj.CRS.from_user_input(crs)

    if len(shape) != 2:
        raise Exception(f"shape is wrong {shape}")

    wms, version = build_wms(wms,version)

    layer = [check_wms_layer(wms,layer=layer)]

    wms_format = check_wms_format(wms,wms_format)

    bounds = bounds.to_crs(crs)

    if style[0] is None:
        img = wms.getmap(layers=layer,
                        srs=str(crs),
                        bbox=tuple(float(i) for i in bounds.total_bounds), #################################################################
                        size=shape,
                        format=wms_format,
                        transparent=transparent
                        )
    else:
        img = wms.getmap(layers=layer,
                        styles=style,
                        srs=str(crs),
                        bbox=tuple(float(i) for i in bounds.total_bounds), ###############################################################
                        size=shape,
                        format=wms_format,
                        transparent=transparent
                        )

    return bites_to_image(img.read()), gpd.GeoSeries(shapely.geometry.box(*bounds.total_bounds),crs=crs)#.to_crs(orig_crs)


def epsg4326_to_tile_coordinates(point:tuple, zoom):
    from math import floor, pi, log, tan, cos

    lon, lat = point
    tileSize = 256
    initialResolution = 2 * pi * 6378137 / tileSize
    originShift = 2 * pi * 6378137 / 2.0

    x = floor((lon + 180) / 360 * (1 << zoom))
    y = floor((1 - log(tan(lat * pi / 180) + 1 / cos(lat * pi / 180)) / pi) / 2 * (1 << zoom))

    return x, y

def request_xyz_tile(url:str ,zoom, point:tuple = (None,None)):
    import requests, copy
    from PIL import Image
    from io import BytesIO

    if "http" not in url:
        url = xyz_service_urls[url]

    if point[0]:
        x,y = epsg4326_to_tile_coordinates(point,zoom)
        url = url.replace("{y}",y)
        url = url.replace("{x}",x)
    elif "{x}" in url or "{y}" in url:
        raise Exception("point is None, Set a point inside desired tile in epsg:4326 coords")
    
    url = url.replace("{z}",zoom)

    response = requests.get(url)

    if response.status_code == 200:
        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")
    
    return img,x,y

def wmts_tile_bbox(tile_matrix,y,x,crs, crs_out="EPSG:4326"):
    from pyproj import Transformer
    # Extract information about the matrix, including tile size and origin
    tile_width = tile_matrix.tilewidth
    tile_height = tile_matrix.tileheight
    origin = tile_matrix.topleftcorner
    scale_denominator = tile_matrix.scaledenominator

    # Create a transformer object for WGS 84
    transformer = Transformer.from_crs(crs, crs_out)

    # Calculate the resolution from the scale denominator
    resolution = 1.0 / scale_denominator

    # Calculate the geographic coordinates (bounding box) covered by the tile
    min_longitude, max_latitude = transformer.transform(origin[0] + x * tile_width * resolution, origin[1] - y * tile_height * resolution)
    max_longitude, min_latitude = transformer.transform(min_longitude + tile_width * resolution, max_latitude - tile_height * resolution)

    return min_longitude, min_latitude, max_longitude, max_latitude
            
def tms_to_geotiff(
    bbox,
    zoom=None,
    resolution=None,
    source="OpenStreetMap",
    crs="EPSG:3857",
    overwrite:bool=False,
    quiet:bool=False,
    check_zoom:bool = False,
    **kwargs,
):
    """Download TMS tiles and convert them to a GeoTIFF. The source is adapted from https://github.com/gumblex/tms2geotiff.
        Credits to the GitHub user @gumblex.

    Args:
        output (str): The output GeoTIFF file.
        bbox (list): The bounding box [minx, miny, maxx, maxy], e.g., [-122.5216, 37.733, -122.3661, 37.8095]
        zoom (int, optional): The map zoom level. Defaults to None.
        resolution (float, optional): The resolution in meters. Defaults to None.
        source (str, optional): The tile source. It can be one of the following: "OPENSTREETMAP", "ROADMAP",
            "SATELLITE", "TERRAIN", "HYBRID", or an HTTP URL. Defaults to "OpenStreetMap".
        crs (str, optional): The output CRS. Defaults to "EPSG:3857".
        overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
        quiet (bool, optional): Suppress output. Defaults to False.
        **kwargs: Additional arguments to pass to gdal.GetDriverByName("GTiff").Create().

    """

    import os
    import io
    import math
    import itertools
    import concurrent.futures

    import numpy
    from PIL import Image

    try:
        True#from osgeo import osr #,gdal
    except ImportError:
        raise ImportError("GDAL is not installed. Install it with pip install GDAL")

    try:
        import httpx

        SESSION = httpx.Client()
    except ImportError:
        import requests

        SESSION = requests.Session()

    #basemaps = get_basemaps()

    if isinstance(source, str):
        if source in xyz_service_urls.keys():
            source = xyz_service_urls[source]
        #elif source in basemaps:
        #    source = basemaps[source]
        elif source.startswith("http"):
            pass
    else:
        raise ValueError(
            f'source must be one of "OpenStreetMap", "ROADMAP", "SATELLITE", "TERRAIN", "HYBRID", or a URL but got {source}'
        )

    if isinstance(bbox, list) and len(bbox) == 4:
        west, south, east, north = bbox
    else:
        raise ValueError(
            "bbox must be a list of 4 coordinates in the format of [xmin, ymin, xmax, ymax]"
        )

    if zoom is None and resolution is None:
        raise ValueError("Either zoom or resolution must be provided")
    elif zoom is not None and resolution is not None:
        raise ValueError("Only one of zoom or resolution can be provided")

    if resolution is not None:
        zoom = resolution_to_zoom(resolution)

    EARTH_EQUATORIAL_RADIUS = 6378137.0

    Image.MAX_IMAGE_PIXELS = None

    #gdal.UseExceptions()
    #web_mercator = osr.SpatialReference()
    #web_mercator.ImportFromEPSG(3857)

    #WKT_3857 = web_mercator.ExportToWkt()

    def from4326_to3857(lat, lon):
        xtile = math.radians(lon) * EARTH_EQUATORIAL_RADIUS
        ytile = (
            math.log(math.tan(math.radians(45 + lat / 2.0))) * EARTH_EQUATORIAL_RADIUS
        )
        return (xtile, ytile)

    def deg2num(lat, lon, zoom):
        lat_r = math.radians(lat)
        n = 2**zoom
        xtile = (lon + 180) / 360 * n
        ytile = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
        return (xtile, ytile)

    def is_empty(im):
        extrema = im.getextrema()
        if len(extrema) >= 3:
            if len(extrema) > 3 and extrema[-1] == (0, 0):
                return True
            for ext in extrema[:3]:
                if ext != (0, 0):
                    return False
            return True
        else:
            return extrema[0] == (0, 0)

    def paste_tile(bigim, base_size, tile, corner_xy, bbox):
        if tile is None:
            return bigim
        im = Image.open(io.BytesIO(tile))
        mode = "RGB" if im.mode == "RGB" else "RGBA"
        size = im.size
        if bigim is None:
            base_size[0] = size[0]
            base_size[1] = size[1]
            newim = Image.new(
                mode, (size[0] * (bbox[2] - bbox[0]), size[1] * (bbox[3] - bbox[1]))
            )
        else:
            newim = bigim

        dx = abs(corner_xy[0] - bbox[0])
        dy = abs(corner_xy[1] - bbox[1])
        xy0 = (size[0] * dx, size[1] * dy)
        if mode == "RGB":
            newim.paste(im, xy0)
        else:
            if im.mode != mode:
                im = im.convert(mode)
            if not is_empty(im):
                newim.paste(im, xy0)
        im.close()
        return newim

    def finish_picture(bigim, base_size, bbox, x0, y0, x1, y1):
        xfrac = x0 - bbox[0]
        yfrac = y0 - bbox[1]
        x2 = round(base_size[0] * xfrac)
        y2 = round(base_size[1] * yfrac)
        imgw = round(base_size[0] * (x1 - x0))
        imgh = round(base_size[1] * (y1 - y0))
        retim = bigim.crop((x2, y2, x2 + imgw, y2 + imgh))
        if retim.mode == "RGBA" and retim.getextrema()[3] == (255, 255):
            retim = retim.convert("RGB")
        bigim.close()
        return retim

    def get_tile(url):
        retry = 3
        while 1:
            try:
                r = SESSION.get(url, timeout=60)
                break
            except Exception:
                retry -= 1
                if not retry:
                    raise
        if r.status_code == 404:
            return None
        elif not r.content:
            return None
        r.raise_for_status()
        return r.content

    def draw_tile(
        source, lat0, lon0, lat1, lon1, zoom, quiet=False, **kwargs
    ):
        bounds = [lon0,lat0,lon1,lat1]
        x0, y0 = deg2num(lat0, lon0, zoom)
        x1, y1 = deg2num(lat1, lon1, zoom)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        corners = tuple(
            itertools.product(
                range(math.floor(x0), math.ceil(x1)),
                range(math.floor(y0), math.ceil(y1)),
            )
        )
        totalnum = len(corners)
        futures = []
        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            for x, y in corners:
                futures.append(
                    executor.submit(get_tile, source.format(z=zoom, x=x, y=y))
                )
            bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
            bigim = None
            base_size = [256, 256]
            for k, (fut, corner_xy) in enumerate(zip(futures, corners), 1):
                bigim = paste_tile(bigim, base_size, fut.result(), corner_xy, bbox)
                if not quiet:
                    print(
                        f"Downloaded image {str(k).zfill(len(str(totalnum)))}/{totalnum}"
                    )

        img = finish_picture(bigim, base_size, bbox, x0, y0, x1, y1)
        return img, bounds
    
    if check_zoom is True:
        while True:
            try:
                image, bounds = draw_tile(
                    source, south, west, north, east, zoom, quiet, **kwargs
                )
                return image, bounds, zoom 
            except:
                zoom -= 1 
                if zoom == -1:
                    raise Exception("Image not available no matter the zoom level")
                
    else:
        try:
            image, bounds = draw_tile(
                source, south, west, north, east, zoom, quiet, **kwargs
            )
            return image, bounds
        except Exception as e:
            raise Exception(e)

def resolution_to_zoom(resolution):
    import math
    """
    Convert map resolution in meters to zoom level for Web Mercator (EPSG:3857) tiles.
    """
    # Web Mercator tile size in meters at zoom level 0
    initial_resolution = 156543.03392804097

    # Calculate the zoom level
    zoom_level = math.log2(initial_resolution / resolution)

    return int(math.ceil(zoom_level))
    
def zoom_to_resolution(zoom):
    # Web Mercator tile size in meters at zoom level 0
    initial_resolution = 156543.03392804097
    resolution = initial_resolution / 2**zoom + 0.001
    return resolution

def get_wtms_max_zoom(wmts,layer:str=None,wmts_format = 'image/jpeg',style:str=None,
                       tilematrixset:str=None,version:str=None,point:gpd.GeoSeries=None):
    from shapely.geometry import Point

    wmts, version = build_wms(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer,version=version)
    wmts_format = check_wmts_format(wmts, layer=layer, wmts_format=wmts_format, version=version)
    style = check_wmts_style(wmts,layer=layer,style=style,version=version)
    tilematrixset = check_wmts_tilematrixset(wmts,layer=layer,tilematrixset=tilematrixset,version=version)
    tile_matrix_crs = wmts.tilematrixsets[tilematrixset].crs
    url = wmts_resource_url(wmts,layer=layer,wmts_format=wmts_format,style=style,tilematrixset=tilematrixset,version=version)

    if point is None:
        x,y = wmts_layer_center(wmts,layer,version)
        point = gpd.GeoSeries(Point([x,y]),crs=4326)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point = point.to_crs(4326).centroid 

    bbox = buffer_in_m(point,0.01)

    zoom = 26

    if "3857" not in tile_matrix_crs:
        raise Exception(f"Your wmts has crs {tile_matrix_crs} but only EPSG:3857 (WebMercator) is accepted")

    _,_, zoom = tms_to_geotiff(list(bbox.to_crs(4326).total_bounds),zoom=zoom,source=url,quiet=True,check_zoom=True)

    resolution = zoom_to_resolution(zoom)
    print(f"Maximum available zoom level is {zoom} for a max resolution of {resolution} m per pixel")
    return zoom, resolution

def get_xyz_max_zoom(url,point:gpd.GeoSeries):
    from shapely.geometry import Point
    if "http" not in url:
        url = xyz_service_urls[url]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = point.to_crs(4326).centroid

    bbox = buffer_in_m(point,0.01)

    zoom = 26

    _,_, zoom = tms_to_geotiff(list(bbox.to_crs(4326).total_bounds),zoom=zoom,source=url,quiet=True,check_zoom=True)

    resolution = zoom_to_resolution(zoom)
    print(f"Maximum available zoom level is {zoom} for a max resolution of {resolution} m per pixel")
    return zoom, resolution

def request_wmts_image(wmts,bounds,shape,layer:str=None,wmts_format = 'image/jpeg',style:str=None,
                       tilematrixset:str=None,version:str=None, max_zoom:int = 999999):

    bounds_utm = bounds.to_crs(bounds.estimate_utm_crs())
    if type(shape) is int:
        dx = (bounds_utm.total_bounds[2] - bounds_utm.total_bounds[0]) ###########################################################
        dy = (bounds_utm.total_bounds[3] - bounds_utm.total_bounds[1])
        if dx < dy:
            shape = (shape, int(dy/dx * shape))
        else:
            shape = (int(dx/dy * shape), shape)
    elif type(shape) is tuple:
        if len(shape) != 2:
            raise Exception(f"shape is wrong {shape}")
    else:
        raise Exception(f"shape should be int or tuple but got {type(shape)}")
    ######################################################################################################################
    
    resolution = min([(bounds_utm.total_bounds[2] - bounds_utm.total_bounds[0]) / shape[0], (bounds_utm.total_bounds[3] - bounds_utm.total_bounds[1]) / shape[1]])

    if resolution_to_zoom(resolution) > max_zoom:
        resolution = zoom_to_resolution(max_zoom)

    wmts, version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer,version=version)
    wmts_format = check_wmts_format(wmts, layer=layer, wmts_format=wmts_format, version=version)
    style = check_wmts_style(wmts,layer=layer,style=style,version=version)
    tilematrixset = check_wmts_tilematrixset(wmts,layer=layer,tilematrixset=tilematrixset,version=version)
    tile_matrix_crs = wmts.tilematrixsets[tilematrixset].crs
    url = wmts_resource_url(wmts,layer=layer,wmts_format=wmts_format,style=style,tilematrixset=tilematrixset,version=version)

    if "3857" not in tile_matrix_crs:
        raise Exception(f"Your wmts has crs {tile_matrix_crs} but only EPSG:3857 (WebMercator) is accepted")

    img,img_bounds = tms_to_geotiff(list(bounds.to_crs(4326).total_bounds),resolution=resolution,source=url,quiet=True)
    img = img.resize(shape) #############################################################################################################################

    img_bounds = gpd.GeoSeries(shapely.geometry.box(*img_bounds),crs=4326)#.to_crs(orig_crs)
    return img, img_bounds

def request_xyz_image(url,bounds,shape, max_zoom:int = 999999):
    if "http" not in url:
        url = xyz_service_urls[url]

    bounds_utm = bounds.to_crs(bounds.estimate_utm_crs())
    if type(shape) is int:
        dx = (bounds_utm.total_bounds[2] - bounds_utm.total_bounds[0]) ###########################################################
        dy = (bounds_utm.total_bounds[3] - bounds_utm.total_bounds[1])
        if dx < dy:
            shape = (shape, int(dy/dx * shape))
        else:
            shape = (int(dx/dy * shape), shape)
    elif type(shape) is tuple:
        if len(shape) != 2:
            raise Exception(f"shape is wrong {shape}")
    else:
        raise Exception(f"shape should be int or tuple but got {type(shape)}")
    ######################################################################################################################
    
    resolution = min([(bounds_utm.total_bounds[2] - bounds_utm.total_bounds[0]) / shape[0], (bounds_utm.total_bounds[3] - bounds_utm.total_bounds[1]) / shape[1]])

    if resolution_to_zoom(resolution) > max_zoom:
        resolution = zoom_to_resolution(max_zoom)

    img,img_bounds = tms_to_geotiff(list(bounds.to_crs(4326).total_bounds),resolution=resolution,source=url,quiet=True)
    img = img.resize(shape) #############################################################################################################################

    img_bounds = gpd.GeoSeries(shapely.geometry.box(*img_bounds),crs=4326)
    return img, img_bounds


def print_wms_layers(wms,version:str=None):
    wms,version = build_wms(wms,version)
    try:
        layers = list(wms.contents)
        return layers
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def print_wmts_layers(wmts,version:str=None):
    wmts,version = build_wmts(wmts,version)
    try:
        layers = list(wmts.contents)
        return layers
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def print_wfs_typenames(wfs,version:str=None):
    wfs,version = build_wfs(wfs,version)
    try:
        typenames = list(wfs.contents)
        return typenames
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def print_wms_methods(wms,version:str=None):
    wms,version = build_wms(wms,version)
    for op in wms.operations:
        print("operation")
        print(op.name)
        print("methods and url")
        print(wms.getOperationByName(op.name).methods)
        print("format options")
        print(wms.getOperationByName(op.name).formatOptions)
        print("")
    return None

def print_wms_format_options(wms,version:str=None):
    wms,version = build_wms(wms,version)
    fo = wms.getOperationByName('GetMap').formatOptions
    for i in fo:
        print(i)
    return fo

def print_wfs_methods(wfs,version:str=None):
    wfs,version = build_wfs(wfs,version)
    for op in wfs.operations:
        print("operation")
        print(op.name)
        print("methods and url")
        print(wfs.getOperationByName(op.name).methods)
        print("format options")
        print(wfs.getOperationByName(op.name).formatOptions)
        print("")
    return None

def print_wmts_format_options(wmts,layer:str = None,version:str=None):
    wmts,version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer)
    fo = wmts[layer].formats
    for i in fo:
        print(i)
    return fo

def print_wfs_format_options(wfs,typename:str = None,version:str=None):
    wfs,version = build_wfs(wfs,version)
    typename = check_wfs_typename(wfs,typename=typename)
    fo = wfs[typename].outputFormats
    #if len(fo) == 0:
    #    fo = wfs.getOperationByName('GetFeature').formatOptions

    for i in fo:
        print(i)
        
    return fo

def print_wmts_styles(wmts,layer:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer)
    try:
        s = list(wmts[layer].styles.keys())
    except:
        s = []

    for i in s:
        print(i)
    return s
    
def print_wfs_styles(wfs,typename:str=None,version:str=None):
    wfs,version = build_wfs(wfs,version)
    typename = check_wfs_typename(wfs,typename=typename)
    try:
        s = list(wfs[typename].styles.keys())
    except:
        s = []

    for i in s:
        print(i)
    return s
    
def print_wfs_crs(wfs,typename:str=None,version:str=None):
    import pyproj
    wfs,version = build_wfs(wfs,version)
    typename = check_wfs_typename(wfs,typename=typename)
    s = [pyproj.CRS.from_user_input(str(i)).to_epsg() for i in wfs[typename].crsOptions]

    for i in s:
        print(i)
    return s

def print_wmts_tilematrixset(wmts,layer:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version)
    layer = check_wmts_layer(wmts,layer=layer)
    t = list(wmts[layer].tilematrixsetlinks.keys())
    for i in t:
        print(i)
    return t

def print_wmts_zooms(wmts,layer:str=None,tilematrixset:str=None,version:str=None):
    wmts,version = build_wmts(wmts,version=version)
    tilematrixset = check_wmts_tilematrixset(wmts,tilematrixset=tilematrixset,layer=layer,version=version)
    z = [i for i in wmts.tilematrixsets[tilematrixset].tilematrix.keys()]
    for i in z:
        print(i)
    return z   