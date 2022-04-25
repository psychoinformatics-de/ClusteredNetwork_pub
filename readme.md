# Spiking attractor model of motor cortex explains modulation of neural and behavioral variability by prior target information

This repository contains all accompanying code to reproduce the figures of the following paper:

*Vahid Rostami, Thomas Rost, Alexa Riehle, Sacha van Albada, Martin Nawrot, Spiking attractor model of motor cortex explains modulation of neural and behavioral variability by prior target information.*

If you use any parts of this code for your own work, please acknowledge us by citing the above paper.


If you have questions or encounter any problems while using this code base, feel free to file a Github issue here and we'll be in touch !

# project layout
This project used Python and Nest for analyses of experimental data and simulations and analyses of the spiking neural networks (SNN). The simulation results (spike trains) are dumped as pickled files.


* `utils/` folder contains all Python scripts to run and analyse both the SNN simulations and experimental data to re-generate the data used for the paper
  * Note1: this requires several TB of disk space and large amount of RAM !
  * Note2: we are working with the NEST developers to include our model as an example in the newest NEST release. We will then adjust the script accordingly and will have a running version of our model using the newest NEST version upon publication.
* `fig_codes/` contains matplotlib scripts to reproduce all the figures of the paper.
* `data/` contains all the experimental and simulation data. To download the data please unzip the following link in the repository: https://www.dropbox.com/s/51atayocaqqha72/data.zip?dl=0
  * NOTE that around 33Gb of memory is needed to unzip the data.
* `enviroment.yml`contains necessary packages to run the codes. Please run the following code to create a conda enviroment:
  * `conda env create -f environment.yml`
  * `conda activate ClusteredNetwork_pub` 

# Reproducing figures
* run the following command inside the `fig_codes`
  * `python figX.py`where X is the number of figure. This will then create figX.pdf or figX.eps or figX.png in the fig_codes folder.
