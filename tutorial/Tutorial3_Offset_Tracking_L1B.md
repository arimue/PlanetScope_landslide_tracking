# Tutorial 3: Offset tracking and DEM building with L1B data

Working with L1B data has the advantage that you as the user gain control over the reference DEM that is used for orthorectification. If you project your data onto a DEM that closely resembles the topography at the time of acquisition of the PlanetScope data, you can greatly reduce the orthorectification error and thus correlate image pairs, even if they were acquired from opposite look directions. This tutorial outlines all steps from data search to velocity calculation for L1B data and also describes how to generate DEMs from unprojected PlanetScope data itself. Many steps are similar to the processing of L3B data, so also refer to tutorials [1](./tutorial/Tutorial1_Data_Search.md) and [2](./tutorial/Tutorial2_Offset_Tracking_L3B.md) for details on these.  

## Step 1: Get some data
