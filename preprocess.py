import glob
from osgeo import gdal, ogr, osr
import numpy as np
import operator
import os
from PIL import Image
import json

class georaster:
    def __init__(self, file_path, reference_crs=None):
        self.file_path = file_path
        gdal.UseExceptions()
        self.dataset = gdal.OpenEx(file_path)
        
        self.crs = osr.SpatialReference()
        self.crs.ImportFromWkt(self.dataset.GetProjection())
        
        self.calculate_extent()
        if reference_crs is None:
            self.E-=self.resX/2.0
            self.W+=self.resX/2.0
            self.N+=self.resY/2.0
            self.S-=self.resY/2.0
        else:
            self.reproject_extent(reference_crs)
        
    def calculate_extent(self):
        geotransform = self.dataset.GetGeoTransform()
        width = self.dataset.RasterXSize
        height = self.dataset.RasterYSize
        
        minx = geotransform[0]
        maxx = geotransform[0] + geotransform[1] * width
        maxy = geotransform[3]
        miny = geotransform[3] + geotransform[5] * height
        
        self.resX=geotransform[1]
        self.resY=geotransform[5]
        self.E = maxx
        self.W = minx
        self.N = maxy
        self.S = miny


    def reproject_extent(self, reference_crs):
        ref_crs = osr.SpatialReference()
        ref_crs.SetFromUserInput(reference_crs)
        
        if not self.crs.IsSame(ref_crs):
            transform = osr.CoordinateTransformation(self.crs, ref_crs)
            
            corners = [(self.W, self.S), (self.E, self.S), (self.W, self.N), (self.E, self.N)]
            
            transformed_corners = [transform.TransformPoint(x, y)[:2] for x, y in corners]

            xs = [point[0] for point in transformed_corners]
            ys = [point[1] for point in transformed_corners]
            self.W = min(xs)
            self.E = max(xs)
            self.S = min(ys)
            self.N = max(ys)

    def export_data_to_binary(self, output_path, metadata_path=None):
        self.data = self.dataset.ReadAsArray()
        self.data.tofile(output_path)
        
        if metadata_path:
            if len(self.data.shape) == 2:
                height, width = self.data.shape
                bands = 1
            else:
                bands, height, width = self.data.shape
            
            metadata = {
                "width": int(width),
                "height": int(height),
                "mid_h": float(np.nanmean(self.data)),
                "h_range": float(np.nanmax(self.data) - np.nanmin(self.data)),
                "dem_resolution_x": float(abs(self.resX)),
                "dem_resolution_y": float(abs(self.resY)),
                "bands": int(bands),
                "west": float(self.W),
                "east": float(self.E),
                "south": float(self.S),
                "north": float(self.N)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

    def export_data_to_png(self, output_path):
        data = self.dataset.ReadAsArray()
        
        # Handle single band vs multi-band (RGB)
        if len(data.shape) == 2:
            # Single band - convert to grayscale
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)
            img = Image.fromarray(data, mode='L')
            img.save(output_path)
        elif len(data.shape) == 3:
            # Multi-band (RGB) - rescale each band independently
            data_uint8 = np.zeros_like(data, dtype=np.uint8)
            for i in range(data.shape[0]):
                band = data[i]
                band_min, band_max = band.min(), band.max()
                if band_max > band_min:
                    data_uint8[i] = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                else:
                    data_uint8[i] = np.zeros_like(band, dtype=np.uint8)
            # Convert from (bands, height, width) to (height, width, bands)
            data_rgb = np.moveaxis(data_uint8, 0, -1)
            img = Image.fromarray(data_rgb, mode='RGB')
            img.save(output_path)

    def horizontal_scale(self):
        return self.W - self.E

    def vertical_scale(self):
        return self.N - self.S

    def get_tiling_and_offset(self,reference_object):
        return reference_object.horizontal_scale() / self.horizontal_scale(), reference_object.vertical_scale() / self.vertical_scale(),(self.W-reference_object.W)/reference_object.horizontal_scale(),(reference_object.S-self.S)/reference_object.vertical_scale()
        

def prepare_for_unity(data_folder="PeaceVallis",outputs_folder="/home/kuba/Unity/VR-GIS/VR-GIS/NewUnityProject/Assets/"):
    outputs_folder = outputs_folder + data_folder
    os.makedirs(outputs_folder, exist_ok=True)
    folder_path="./Projects/" + data_folder
    DEM_file=folder_path + "/DEM.tif"
    texture_files = glob.glob(folder_path + "/Textures/*.tif")
    dem = georaster(DEM_file)
    dem.export_data_to_binary(outputs_folder + "/dem.bytes", outputs_folder + "/dem_metadata.json")
    tiling_offset_array=np.zeros((len(texture_files), 4))
    for i, texture_file in enumerate(texture_files):
        texture = georaster(texture_file, reference_crs=dem.crs.ExportToWkt())
        tiling_offset_array[i] = texture.get_tiling_and_offset(dem)
        texture.export_data_to_png(outputs_folder + "/texture_" + str(i) + ".png")
    tiling_offset_array.tofile(outputs_folder + "/tiling_offset.bytes")

if __name__ == "__main__":
    prepare_for_unity()
