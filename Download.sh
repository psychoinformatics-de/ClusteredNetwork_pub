#!/bin/bash

# check if data folder exists and create it if not
# if it exists, check if it is empty, if not ask if it should be overwritten, if not exit
if [ ! -d data ]; then
    mkdir data
else
    if [ "$(ls -A data)" ]; then
        read -p "The data folder is not empty. Do you want to overwrite it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf data
            mkdir data
        else
            exit 1
        fi
    fi
fi


# create tmp folder to store the downloaded files
if [ ! -d tmp ]; then
    mkdir tmp
fi

cd tmp
# download the data
git clone https://gin.g-node.org/nawrotlab/EI_clustered_network
cd EI_clustered_network
git annex get *

# move the data to the data folder
mv * ../../data
cd ../../
rm -rf tmp
chmod -R ugo+rwX data/




