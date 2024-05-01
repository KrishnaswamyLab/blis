# BLIS-Net: Classifying and Analyzing Signals on Graphs
BLIS-Net (bi-Lipschitz Scattering Network) is a provably powerful GNN for graph signal classification. 

TODO: add figure

## Introduction

BLIS-Net's core module computes BLIS-scattering features on each node and hence may be flexibly incorporated into GNNs for a variety of downstream tasks.   
TODO: explain code layout

## Installation

Create a conda environment

~~~
conda create -n blis python=3.9`
conda activate blis
cd blis
pip install -e .
~~~

### Data download (optional)
The data used in the paper may be downloaded from the following [link](https://drive.google.com/file/d/1zMItIcmXFbN66sEZOPql30dKgxPFo5_v/view?usp=sharing).
Please download the zip into the main project directory data directory, perhaps following something like:
~~~
rm -rf data
unzip data.zip
mv data_export data
rm data.zip
~~~


