# LOD2CREATOR

## Task

### Model-driven approach for 3D roof reconstruction

#### Input

• Segmented roof point clouds (30pts)
• Building footprints from FKB or OSM

#### Output

• LoD2 building models

## Description

• Define rules of geometric structures on building roofs
• Projecting segments of point clouds on ground
• Use edges of building footprints as edges of some segments
• Find edges that are parallel to horizontal plan
• Iterative methods to find intersection points of segments
• Linking edges into polygons of segments
• Writing 3D models in CityGML format
• Evaluation 

# To use this progject

## Install environment
conda env create -f environment.yml

## Activate environment
conda activate lod2creator

## Add changes after installing packages (Add pip packages manually)
conda env export --from-history > environment.yml


## Update environment after pulling changes
conda activate lod2creator
conda env update -f environment.yml --prune
