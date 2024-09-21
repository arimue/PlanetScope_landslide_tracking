# Landslide Tracking with PlanetScope Data 

This repository provides scripts and processing examples for tracking landslides (or other Earth surface processes) based on optical [PlanetScope](https://developers.planet.com/docs/data/planetscope/) data using image cross-correlation. 

## Prerequisites
### Linux OS
The scripts in this repository rely on functionalities provided by the [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/index.html). ASP builds are available for Linux and macOS. If you are using Windows, you will need to install a Windows Subsystem for Linux in order to run the ASP tools.
### Python
The provided code is written in Python and consists of modules that you can import and use in your own Python scripts.
### [Planet](https://www.planet.com/) Account
To search for and access PlanetScope data via the API, you will need to authenticate with your Planet account. You can find more information on how to get started with Planet [here](https://www.planet.com/get-started/). If you are a university student, faculty member or researcher, you may apply to the Planet [Education and Research Program](https://www.planet.com/markets/education-and-research/).

## Installation 

### Step 1: 
Clone the PlanetScope_landslide_tracking repository: `git clone https://github.com/UP-RS-ESP/PlanetScope_landslide_tracking`.
### Step 2: 
Enter the cloned repository `cd PlanetScope_landslide_tracking` and use [conda](https://conda.io/projects/conda/en/latest/index.html) to create a new environment and install all necessary Python packages using the given environment.yml file: `conda env create -f environment.yml` 
Then activate the newly created environment: `conda activate planet`
### Step 3:
Install [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/index.html). Download the latest stable release from [GitHub](https://github.com/NeoGeographyToolkit/StereoPipeline/releases) and unzip it. Detailed instructions can be found in the [ASP Documentation](https://stereopipeline.readthedocs.io/en/latest/installation.html).
### Step 4: 
Install the [Planet SDK for Python](https://github.com/planetlabs/planet-client-python): `pip install planet` (see [Documentation](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/get-started/quick-start-guide/#step-2-install-the-planet-sdk-for-python))

You will need to authenticate with your Planet account. Open a terminal and enter the following command: `planet auth init`. Then enter your user name and password. For more information refer to the [Planet SDK Documentation](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/get-started/quick-start-guide/#step-4-sign-on-to-your-account).

### Step 5 (optional):
Video and GIF generation (timelapse) relies on [FFmpeg](https://www.ffmpeg.org/). On Linux, you can install FFmpeg with the apt package manager:
```bash
sudo apt update
sudo apt install ffmpeg
```

## Processing examples

Detailed instructions on how to use the provided tools are stored in the [tutorial](./tutorial) folder:

[Tutorial 1](./tutorial/Tutorial1_Data_Search.md): Searching for PlanetScope data\
[Tutorial 2](./tutorial/Tutorial2_Offset_Tracking_L3B.md): Offset tracking with L3B data\
[Tutorial 3](./tutorial/Tutorial3_Offset_Tracking_L1B.md): Offset tracking with L1B data\
[Tutorial 4](./tutorial/Tutorial4_DEM_Building.md): DEM building with PlanetScope L1B data and Ames Stereo Pipeline\
[Tutorial 5](./tutorial/Tutorial5_Remapping_and_Video.md): Remapping and timelapse generation

[example.py](./tutorial/example.py): An exemplary Python script

## Citation

Mueting, A. and Bookhagen, B.: Tracking slow-moving landslides with PlanetScope data: new perspectives on the satellite’s perspective, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-1698, 2023.
