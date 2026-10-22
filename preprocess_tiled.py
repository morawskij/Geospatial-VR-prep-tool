import glob
from osgeo import gdal, ogr, osr
import numpy as np
import operator
import os
from PIL import Image
import json
from tqdm import tqdm
from scipy.ndimage import zoom

class georaster:
    def __init__(self, file_path):
        self.file_path = file_path
        gdal.UseExceptions()
        self.dataset = gdal.OpenEx(file_path)
        
        self.crs = osr.SpatialReference()
        self.crs.ImportFromWkt(self.dataset.GetProjection())
        
        self.calculate_extent()

        self.data = self.dataset.ReadAsArray()

    def calculate_extent(self):
        geotransform = self.dataset.GetGeoTransform()
        width = self.dataset.RasterXSize
        height = self.dataset.RasterYSize
        
        minx = geotransform[0]
        maxx = geotransform[0] + geotransform[1] * width
        maxy = geotransform[3]
        miny = geotransform[3] + geotransform[5] * height
        
        self.resX=abs(geotransform[1])
        self.resY=abs(geotransform[5])
        self.extent = Extent(minx, maxx, miny, maxy)

    def reproject_extent(self, reference_crs):
        ref_crs = osr.SpatialReference()
        ref_crs.SetFromUserInput(reference_crs)
        
        if not self.crs.IsSame(ref_crs):
            transform = osr.CoordinateTransformation(self.crs, ref_crs)
            
            corners = [(self.extent.W, self.extent.S), (self.extent.E, self.extent.S), (self.extent.W, self.extent.N), (self.extent.E, self.extent.N)]
            
            transformed_corners = [transform.TransformPoint(x, y)[:2] for x, y in corners]

            xs = [point[0] for point in transformed_corners]
            ys = [point[1] for point in transformed_corners]
            self.extent.W = min(xs)
            self.extent.E = max(xs)
            self.extent.S = min(ys)
            self.extent.N = max(ys)

    def horizontal_scale(self):
        return self.extent.W - self.extent.E

    def vertical_scale(self):
        return self.extent.N - self.extent.S

    def get_tiling_and_offset(self,reference_object):
        return reference_object.horizontal_scale() / self.horizontal_scale(), reference_object.vertical_scale() / self.vertical_scale(),(self.extent.W-reference_object.extent.W)/reference_object.horizontal_scale(),(reference_object.extent.S-self.extent.S)/reference_object.vertical_scale()
        
class DEM (georaster):
    def __init__(self, file_path, export_folder):
        super().__init__(file_path)
        """
        self.E-=self.resX/2.0
        self.W+=self.resX/2.0
        self.N+=self.resY/2.0
        self.S-=self.resY/2.0
        """
        self.x, self.y = self.data.shape      
        self.pyramid_levels = int(np.ceil(max(0,np.log2(max(self.x/256, self.y/256)))+1))
        self.pyramid_extent = Extent(self.extent.W, self.extent.W+self.resX*2**(self.pyramid_levels+7) , self.extent.S, self.extent.S+self.resY*2**(self.pyramid_levels+7))
        self.export_folder=export_folder

    def prepare(self):
        self.generate_tiles(self.export_folder)
        self.export_metadata(self.export_folder + "/dem_metadata.json")

    def generate_tiles(self,export_folder):
        for level in range(self.pyramid_levels):
            print(f"Generating level {level}")
            step = 2**(self.pyramid_levels-level-1)
            os.makedirs(export_folder + f"/{level}", exist_ok=True)
            for x in range(int(1+self.x/(step*256))):
                for y in range(int(1+self.y/(step*256))):
                    temp_tile = tile(export_folder,level,y,x)
                    max_x=min(self.x//step,(x+1)*256)
                    max_y=min(self.y//step,(y+1)*256)
                    temp_tile.data[:max_x-x*256,:max_y-y*256] = self.data[x*256*step:max_x*step:step,y*256*step:max_y*step:step]
                    temp_tile.export()

    def export_metadata(self,metadata_path):
            metadata = {
                "width": int(256),
                "height": int(256),
                "mid_h": float(np.nanmean(self.data)),
                "h_range": float(np.nanmax(self.data) - np.nanmin(self.data)),
                "pyramid_levels": int(self.pyramid_levels),
                "dem_resolution_x": float(abs(self.resX)),
                "dem_resolution_y": float(abs(self.resY)),
                "west": float(self.extent.W),
                "east": float(self.extent.E),
                "south": float(self.extent.S),
                "north": float(self.extent.N)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

class tile:
    def __init__(self,export_folder,level,x,y):
        self.data = np.full((256,256), -9999, dtype=np.float32)
        self.export_folder = export_folder
        self.level = level
        self.x = x
        self.y = y

    def export(self):
        output_path = self.export_folder + "/" + str(self.level) + "/" + str(self.x) + "_" + str(self.y) + ".bytes"
        self.data.tofile(output_path)            

class Extent:
    def __init__(self,W,E,S,N):
        self.W = W
        self.E = E
        self.S = S
        self.N = N

    def __str__(self):
        return "W: " + str(self.W) + "E: " + str(self.E) + "S: " + str(self.S) + "N: " + str(self.N)

    def check_overlap(self,other_extent):
        return self.W < other_extent.E and self.E > other_extent.W and self.S < other_extent.N and self.N > other_extent.S

    def get_overlap(self,other_extent):
        return Extent(
            max(self.W, other_extent.W),
            min(self.E, other_extent.E),
            max(self.S, other_extent.S),
            min(self.N, other_extent.N)
        )

class Texture (georaster):
    def __init__(self, file_path,reference_dem):
        super().__init__(file_path)
        self.reference_dem = reference_dem
        self.reproject_extent(self.reference_dem.crs.ExportToWkt())
        
        # Convert to uint8 and rescale to 0-255 range once at initialization
        if len(self.data.shape) == 2:
            data_min, data_max = self.data.min(), self.data.max()
            if data_max > data_min:
                self.data = ((self.data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                self.data = np.zeros_like(self.data, dtype=np.uint8)
        elif len(self.data.shape) == 3:
            data_uint8 = np.zeros_like(self.data, dtype=np.uint8)
            for i in range(self.data.shape[0]):
                band = self.data[i]
                band_min, band_max = band.min(), band.max()
                if band_max > band_min:
                    data_uint8[i] = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                else:
                    data_uint8[i] = np.zeros_like(band, dtype=np.uint8)
            self.data = data_uint8

    def export_cropped(self, output_path, crop_extent, window_size=(256, 256), nodata_value=0):
        """
        Export texture cropped to specified extent, resampled to window size.
        Areas outside the texture extent are filled with nodata_value.
        
        Args:
            output_path: Path to save the output image
            crop_extent: Extent object defining the crop area (W, E, S, N)
            window_size: Tuple (width, height) for output image size
            nodata_value: Value to use for areas outside texture extent
        """
        if crop_extent.check_overlap(self.extent):
            intersect = crop_extent.get_overlap(self.extent)
            if len(self.data.shape) == 2:
                output_data = np.full(window_size[::-1], nodata_value, dtype=np.uint8)
            else:
                output_data = np.full((self.data.shape[0], window_size[1], window_size[0]), nodata_value, dtype=np.uint8)

            # Convert geographic coordinates to pixel coordinates
            px_W = int((intersect.W - self.extent.W) / self.resX)
            px_E = int((intersect.E - self.extent.W) / self.resX)
            px_S = int((self.extent.N - intersect.S) / self.resY)
            px_N = int((self.extent.N - intersect.N) / self.resY)
            
            # Clamp to valid pixel range
            px_W = max(0, min(px_W, self.dataset.RasterXSize))
            px_E = max(0, min(px_E, self.dataset.RasterXSize))
            px_S = max(0, min(px_S, self.dataset.RasterYSize))
            px_N = max(0, min(px_N, self.dataset.RasterYSize))
            
            # Extract the intersecting region
            if len(self.data.shape) == 2:
                cropped_data = self.data[px_N:px_S, px_W:px_E]
            else:
                cropped_data = self.data[:, px_N:px_S, px_W:px_E]
            
            # Calculate where this intersection should be placed in the output window
            crop_width = crop_extent.E - crop_extent.W
            crop_height = crop_extent.N - crop_extent.S
            
            # Calculate proportional position in output window
            out_px_W = int(((intersect.W - crop_extent.W) / crop_width) * window_size[0])
            out_px_E = int(((intersect.E - crop_extent.W) / crop_width) * window_size[0])
            out_px_S = int(((crop_extent.N - intersect.N) / crop_height) * window_size[1])
            out_px_N = int(((crop_extent.N - intersect.S) / crop_height) * window_size[1])

            # Resample cropped data to fit the output region
            if cropped_data.size > 0:
                target_shape = (out_px_N - out_px_S, out_px_E - out_px_W) if len(self.data.shape) == 2 else (self.data.shape[0], out_px_N - out_px_S, out_px_E - out_px_W)
                
                if len(self.data.shape) == 2:
                    zoom_factors = (target_shape[0] / cropped_data.shape[0], target_shape[1] / cropped_data.shape[1])
                else:
                    zoom_factors = (1, target_shape[1] / cropped_data.shape[1], target_shape[2] / cropped_data.shape[2])
                
                resampled = zoom(cropped_data.astype(np.float32), zoom_factors, order=1, mode='constant', cval=nodata_value).astype(np.uint8)
                
                # Place resampled data in output window
                if len(self.data.shape) == 2:
                    output_data[out_px_S:out_px_N, out_px_W:out_px_E] = resampled
                else:
                    output_data[:, out_px_S:out_px_N, out_px_W:out_px_E] = resampled
        
            # Save as image
            if len(output_data.shape) == 2:
                img = Image.fromarray(output_data, mode='L')
            else:
                data_rgb = np.moveaxis(output_data, 0, -1)
                img = Image.fromarray(data_rgb, mode='RGB')
            img.save(output_path)
            return min(zoom_factors)<1
        else:
            return False

class TextureFamily:
    def __init__(self,texture_file_list, reference_dem, export_folder):
        self.textures = []
        self.reference_dem = reference_dem
        for texture_file in texture_file_list:
            self.textures.append(Texture(texture_file, self.reference_dem))
        self.export_folder = export_folder

    def recursive_pyramid_generation(self,level=0,x=0,y=0):
        os.makedirs(self.export_folder + f"/{level}", exist_ok=True)
        alpha1 = x/2**level
        alpha2 = alpha1+1/2**level
        beta1 = y/2**level
        beta2 = beta1+1/2**level
        W = self.reference_dem.extent.W*(1-alpha1)+self.reference_dem.pyramid_extent.E*alpha1
        E = self.reference_dem.extent.W*(1-alpha2)+self.reference_dem.pyramid_extent.E*alpha2
        S = self.reference_dem.extent.S*(1-beta1)+self.reference_dem.pyramid_extent.N*beta1
        N = self.reference_dem.extent.S*(1-beta2)+self.reference_dem.pyramid_extent.N*beta2
        progression_needed = False
        cropping_extent=Extent(W,E,S,N)
        for i, tex in enumerate(self.textures):
            new_flag = tex.export_cropped(self.export_folder+ "/" + str(level) + "/" + str(x) + "_" + str(y) + "_"+ str(i) + ".png",cropping_extent)
            progression_needed = progression_needed or new_flag           
        if progression_needed:
            for i in range(2):
                for j in range(2):
                    self.recursive_pyramid_generation(level=level+1,x=2*x+i,y=2*y+j)
    



def prepare_for_unity(data_folder="PeaceVallis",outputs_folder="/home/kuba/Unity/VR-GIS/VR-GIS/NewUnityProject/Assets/StreamingAssets/Projects/"):
    outputs_folder = outputs_folder + data_folder
    os.makedirs(outputs_folder, exist_ok=True)
    folder_path="./Projects/" + data_folder
    DEM_file=folder_path + "/DEM.tif"
    texture_files = glob.glob(folder_path + "/Textures/*.tif")
    dem = DEM(DEM_file, outputs_folder)
    dem.prepare()
    textures = TextureFamily(texture_files,dem,outputs_folder)
    textures.recursive_pyramid_generation()

if __name__ == "__main__":
    prepare_for_unity(data_folder="PeaceVallis")
