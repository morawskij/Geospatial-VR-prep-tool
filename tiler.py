from pathlib import Path
import hashlib
from turtle import window_height, window_width
import rasterio
import numpy as np
import cv2
import os

basedir = os.path.dirname(__file__)

def scale_tile(tile, scales):
    # tile shape: (bands, H, W)
    out = np.empty_like(tile, dtype=np.float32)

    for b, (lo, hi) in enumerate(scales):
        band = tile[b].astype(np.float32)
        band = (band - lo) / (hi - lo + 1e-6)
        band = np.clip(band, 0, 1)
        out[b] = band * 255.0

    return out.astype(np.uint8)

def hirise_scaling(src):
    return [(0,1023)]

def hirise_color_scaling(src):
    return [(0,1023),(0,1023),(0,1023)]

def fake_scaling(src):
    return [(0,255),(0,255),(0,255)]

def estimate_band_scaling(src, num_samples=100, sample_size=16):

    w, h = src.width, src.height
    bands = src.count
    samples = [[] for _ in range(bands)]

    for _ in range(num_samples):
        x = np.random.randint(0, max(0, w - sample_size))
        y = np.random.randint(0, max(0, h - sample_size))
        window = rasterio.windows.Window(x, y, sample_size, sample_size)
        data = src.read(window=window)
        data = np.where(data==src.nodata, np.nan, data)

        for b in range(bands):
            samples[b].append(data[b].ravel())

    scales = []
    for b in range(bands):
        vals = np.concatenate(samples[b])
        scales.append((np.nanmin(vals),np.nanmax(vals)))

    return scales

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def check_complete(dir,total):
    count=0
    for root, dirs, files in os.walk(dir):
        count+=len(files)
    return count == total

def number_of_tiles(path,tile_size=512):
    with rasterio.open(path) as src:
        w=src.width
        h=src.height
        bb=int(np.ceil(src.count/3))
        out_dir = os.path.join(basedir,"tiles",file_hash(path))
        num_lvls = int(np.ceil(np.log2(max(np.ceil(w/tile_size), np.ceil(h/tile_size))))) + 1
        total = 0
        for _ in range(num_lvls):
            tiles_x = int(np.ceil(w / tile_size))
            tiles_y = int(np.ceil(h / tile_size))
            total += tiles_x * tiles_y*bb
            w = max(1, w // 2)
            h = max(1, h // 2)
        if os.path.exists(out_dir):
            if check_complete(out_dir,total):
                return total, num_lvls, out_dir, False, src.width, src.height, src.count, src.profile["transform"][0]
        return total, num_lvls, out_dir, True, src.width, src.height, src.count, src.profile["transform"][0]

def read_tile(src,window,out_shape):

    tile = np.zeros(out_shape)

    if window.width * window.height * out_shape[0] > 10000000000:

        tile[:,:int(np.ceil(out_shape[1]/2)),:int(np.ceil(out_shape[2]/2))]=read_tile(src,rasterio.windows.Window(window.col_off, window.row_off, int(np.ceil(window.width/2)),int(np.ceil(window.height/2))),(out_shape[0],int(np.ceil(out_shape[1]/2)),int(np.ceil(out_shape[2]/2))))
        tile[:,:int(np.ceil(out_shape[1]/2)),int(np.ceil(out_shape[2]/2)):]=read_tile(src,rasterio.windows.Window(window.col_off+int(np.ceil(window.width/2)), window.row_off, window.width-int(np.ceil(window.width/2)),int(np.ceil(window.height/2))),(out_shape[0],int(np.ceil(out_shape[1]/2)),out_shape[2]-int(np.ceil(out_shape[2]/2))))
        tile[:,int(np.ceil(out_shape[1]/2)):,:int(np.ceil(out_shape[2]/2))]=read_tile(src,rasterio.windows.Window(window.col_off, window.row_off+int(np.ceil(window.height/2)), int(np.ceil(window.width/2)),window.height-int(np.ceil(window.height/2))),(out_shape[0],out_shape[1]-int(np.ceil(out_shape[1]/2)),int(np.ceil(out_shape[2]/2))))
        tile[:,int(np.ceil(out_shape[1]/2)):,int(np.ceil(out_shape[2]/2)):]=read_tile(src,rasterio.windows.Window(window.col_off+int(np.ceil(window.width/2)), window.row_off+int(np.ceil(window.height/2)),window.width-int(np.ceil(window.width/2)),window.height-int(np.ceil(window.height/2))),(out_shape[0],out_shape[1]-int(np.ceil(out_shape[1]/2)),out_shape[2]-int(np.ceil(out_shape[2]/2))))

    else:
        try:
            tile=src.read(window=window, out_shape=out_shape)
        except:
            pass

    return tile    

def generate_tiles(path,num_lvls,out_dir,w,h,tile_size=512):
    if str(path)[-3:]=="jpg":
        standard_image=True
    else:
        standard_image=False
    with rasterio.open(path) as src:
        bands=src.count
        if bands==1:
            if str(path)[-3:]=="tif":
                scales = estimate_band_scaling(src)
            else:
                scales = hirise_scaling(src)
        else:
            if standard_image:
                scales = fake_scaling(src)
            elif "sentinel" in str(path):
                scales = estimate_band_scaling(src)
            else:
                scales = hirise_color_scaling(src)
        ts=tile_size*2**(num_lvls-1) 
        existing=0
        for i in range(num_lvls):
            lvl_dir = os.path.join(out_dir,str(i))
            Path(lvl_dir).mkdir(parents=True, exist_ok=True)
            true_ind_x = 0
            true_ind_y = 0
            ind_x=0
            ind_y=0
            while true_ind_y < h:
                while true_ind_x < w:
                    window = rasterio.windows.Window(true_ind_x, true_ind_y, ts, ts)
                    window_full = rasterio.windows.Window(0, 0, w, h)
                    valid_window=window.intersection(window_full)
                    out_shape=(bands, int(valid_window.height*tile_size/ts), int(valid_window.width*tile_size/ts))
                    tile = read_tile(src,valid_window,out_shape)
                    j=0
                    k=0
                    while j+3 <= bands:
                        out_shape=(3, int(valid_window.height*tile_size/ts), int(valid_window.width*tile_size/ts))
                        final_tile = np.zeros((3, tile_size, tile_size))
                        final_tile[:, :out_shape[1], :out_shape[2]] = tile[j:j+3,:,:]
                        final_tile=scale_tile(final_tile,scales[j:j+3])
                        cv2.imwrite(os.path.join(lvl_dir,"{}_{}_{}.jpg".format(ind_x, ind_y,k)), np.transpose(final_tile, (1, 2, 0))[:,:,::-1])
                        j+=3
                        k+=1
                    if bands-j>0:
                        out_shape=(1+2*int(bands-j==2), int(valid_window.height*tile_size/ts), int(valid_window.width*tile_size/ts))
                        final_tile = np.zeros((1+2*int(bands-j==2), tile_size, tile_size))
                        final_tile[:bands-j, :out_shape[1], :out_shape[2]] = tile[j:j+bands-j,:,:]
                        final_tile=scale_tile(final_tile,scales[j:bands])
                        cv2.imwrite(os.path.join(lvl_dir,"{}_{}_{}.jpg".format(ind_x, ind_y,k)), np.transpose(final_tile, (1, 2, 0)))
                    true_ind_x += ts
                    ind_x+=1
                    existing +=k+1
                    sys.stdout.write("Progress:{}".format(str(existing)))
                    sys.stdout.flush()
                ind_y += 1
                true_ind_y += ts
                ind_x = 0
                true_ind_x = 0
            ts//=2
        return

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    num_lvls = int(sys.argv[2])
    out_dir = sys.argv[3]
    w = int(sys.argv[4])
    h = int(sys.argv[5])
    generate_tiles(path, num_lvls, out_dir,w,h)
