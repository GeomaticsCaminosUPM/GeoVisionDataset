from setuptools import setup, find_packages

setup(
    name="geodataset",
    version="0.1.0",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Miguel Ureña Pliego",
    author_email="miguel.urena@upm.es",
    url="https://github.com/GeomaticsCaminosUPM/geodataset",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "geopandas>=1.0.0",
        "pandas>=2.0.0",
        "shapely>=2.0.0",
        "numpy>=2.0.0",
        "matplotlib>=3.0.0",
        "pycocotools>=2.0.0",
        "folium>=0.16.0",
        "opencv-python>=2.0.0",
        "pillow>=6.0.0",
        "owslib>=0.32.0",
        "pyproj>=3.0.0",
        "ipyleaflet>=0.19.0",
        "osm2geojson>=0.1.27",
        "rasterio>=1.3.0",
        "mapclassify>=2.0.0",
        "leafmap>=0.20.0",
        "regex>=2.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
