# Spiking Attractor Model of Motor Cortex: Modulating Neural and Behavioral Variability via Prior Target Information

This repository hosts the code necessary to reproduce the figures presented in our paper:

**Vahid Rostami, Thomas Rost, Felix Schmitt, Alexa Riehle, Sacha van Albada, Martin Nawrot. "Spiking attractor model of motor cortex explains modulation of neural and behavioral variability by prior target information."**

Should you utilize any segments of this code for your work, kindly acknowledge us by citing the above paper.

Should you encounter any inquiries or issues while using this codebase, please create a GitHub issue, and we'll promptly engage with you!

## Project Structure
This project utilizes Python and Nest for analyzing experimental data, simulating spiking neural networks (SNN), and producing figures. The simulation results (spike trains) are stored as pickled files.

* `fig_codes/` houses matplotlib scripts to replicate all paper figures.
* `data/` includes all experimental and simulated data required to reproduce the figures. The data is hosted on G-Node GIN and can be downloaded using the instructions below.
* `utils/` contains Python scripts for executing SNN simulations and analyzing simulated/experimental data.

## Environment Setup
The `environment.yml` file contains necessary packages to execute the code. To create a Conda environment:

```bash
conda env create -f environment.yml
conda activate ClusteredNetwork_pub
```
Unfortunately, the conda environment does not contain all dependencies. You might need to install some libaries outside of the conda environment.
We recommend using our docker image to run the code. Please see the section below for more information.


## Reproducing figures
To recreate specific figures, execute the following command within the fig_codes directory:
```bash
python figX.py
```
Replace **'X'** with the figure number. This will generate **'figX.pdf'**, or **'figX.png'** within the **'fig_codes'** folder.

## Accessing Data
Experimental data and simulation results are also available on G-Node GIN in the repository 
[nawrotlab/EI_clustered_network](https://gin.g-node.org/nawrotlab/EI_clustered_network).
This repository is roughly 16GB in size. To download the data via the web interface, 
follow the instructions on the provided link. We recommend using the command line and 
[git-annex](https://git-annex.branchable.com/install/) for downloads:

```bash
git clone https://gin.g-node.org/nawrotlab/EI_clustered_network
cd EI_clustered_network
git annex get *
```
Alternatively, utilize the provided script to download the data:
```bash
./Download.sh
```
This script creates a ***'data'*** folder in the repository and initiates the download. 
It verifies if the data is already present and skips download if it exists. 
Note that the download may take a considerable amount of time, and git-annex might appear stalled but will resume eventually.

## Docker Image for Code Execution
We provide a docker image, 
[fschmitt/clustered_network_pub:nest2_20](https://hub.docker.com/repository/docker/fschmitt/clustered_network_pub/), 
to run the code.
The image is accessed via Docker Hub. To use:

```bash 
docker pull fschmitt/clustered_network_pub:nest2_20
docker run -d   -it   --name clusternet   --mount type=bind,source="$(pwd)"/ClusteredNetwork_pub,target=/app   fschmitt/clustered_network_pub:nest2_20
docker exec -it clusternet /bin/bash
```

Once inside the container, execute the download script or run the code as previously described. To exit the container:
```bash
exit
docker stop clusternet
docker rm clusternet
```

If you prefer not to mount the repository into the docker image, you can clone it inside the container:
```bash
git clone https://github.com/nawrotlab/ClusteredNetwork_pub.git
cd ClusteredNetwork_pub
```
### Known problems of Docker
Older docker version might not automatically set up a functioning network bridge. The download_data.sh script will thus not be able to access the internet and fail.
You can circumvent this by creating a bridge manually:
```bash
docker network create --driver bridge common
docker run -d -it --network common --name clusternet --mount type=bind,source="$(pwd)"/ClusteredNetwork_pub,target=/app fschmitt/clustered_network_pub:nest2_20
```


