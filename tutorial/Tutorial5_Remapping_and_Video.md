# Tutorial 5: Remapping and timelapse generation

Besides correcting disparity maps, the polynomial fit can also be used to remap secondary images to a common reference and obtain more precisely coregistered images. These can for example be used to create a timelapse of the monitored landslide. 

## Step 1: Matchmaking

As a first step, generate a matchfile that matches all scenes in a directory to a common reference. Use the `match_to_one_ref()` function from the preprocessing tools for that purpose. This will take all images from one folder and match it with the oldest acquisition. 

``` python
import preprocessing_functions as preprocessing

work_dir = "./tutorial/L3B"
matches = preprocessing.match_to_one_ref(work_dir)
```

## Step 2: Apply polynomial fit and remap

To remap the secondary images according to the polynomial fit and thus improve the co-registration between reference and secondary image, again use the `apply_polyfit()` function and make sure to set the `save_remapped_sec` option to `True`. After disparity map generation (if not already present) and polynomial fitting, the pixels in the secondary image will be shifted according to the disparity fit and then interpolated using bilinear interpolation. All remapped secondary images will be saved in your working directory with the file extension `*_remap.tif`.

``` python
import optimization_functions as opt

dmaps_pfit = opt.apply_polyfit(matches, prefix_ext= "_L3B", order = 2, demname = cop_dem, save_remapped_sec = True)
```

## Step 3: Create video

From the remapped images (or also original PlanetScope scenes) you can create a nice timelapse of the landslide motion:

``` python
import postprocessing_functions as postprocessing

postprocessing.make_video(matches, video_name = "timelapse_remapped.mp4", ext = "_remap", crop = 300)
```

This function will find the reference and remapped secondary images and create a video (timelapse_remapped.mp4) and a GIF (timelapse_remapped.gif) out of these images. Use the crop options to discard pixels along the image margins. If you would like to create a timelapse from the original PlanetScope scenes, set the file extension to "".

<img src='./figures/remapped_scenes.gif' width='500'>
Images © 2023 Planet Labs PBC
