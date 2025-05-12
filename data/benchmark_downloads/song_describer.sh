#!/bin/bash

cd /data2/rsalgani/song_describer/

# # # Set the Zenodo record id number
RECORD_ID="10072001"
# # download main dataset file "song_describer.csv"
# curl -L -O "https://zenodo.org/record/${RECORD_ID}/files/song_describer.csv"

# download audio data (> 3GB)
curl -L -O "https://zenodo.org/record/${RECORD_ID}/files/audio.zip"
unzip audio.zip
rm audio.zip