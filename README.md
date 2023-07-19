# Landslide Tracking with PlanetScope Data 

This repository provides scripts and processing examples for tracking landslides (or other Earth surface processes) based on optical [PlanetScope](https://developers.planet.com/docs/data/planetscope/) data using image-cross correlation. 

## Prerequisites
### Linux OS
The scripts in this repository rely on functionalities provided by the [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/index.html). ASP builds are available for Linux and macOS. If you are using Windows, you will need to install a Windows Subsystem for Linux in order to run the ASP tools.
### [Planet](https://www.planet.com/) Account
To search for and access PlanetScope data via the API, you will need to authenticate with your Planet account. You can find more information on how to get started with Planet [here](https://www.planet.com/get-started/). If you are a university student, faculty member or researcher, you may apply to the Planet [Education and Research Program](https://www.planet.com/markets/education-and-research/).
## Installation 

### Step 1: 
Clone the PlanetScope_landslide_tracking repository: `git clone https://github.com/UP-RS-ESP/PlanetScope_landslide_tracking`.
### Step 2: 
Enter the cloned repository `cd PlanetScope_landslide_tracking` and use [conda](https://conda.io/projects/conda/en/latest/index.html) to create a new environment and install all necessary Python packages using the given environment.yml file: `conda env create -f environment.yml` 
Then activate the newly created environment: `conda activate planet`
### Step 3: 
Install [Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/index.html). Download the latest stable release from [GitHub](https://github.com/NeoGeographyToolkit/StereoPipeline/releases) and unzip it. Detailed instructions can be found in the [ASP](https://stereopipeline.readthedocs.io/en/latest/installation.html)

## Setup

## Processing examples

## Citation

A. Mueting and B. Bookhagen: Tracking slow-moving landslides with PlanetScope data: new perspectives on the satellite's perspective. (in prep.)
