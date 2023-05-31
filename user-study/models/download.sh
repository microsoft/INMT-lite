#!/bin/bash

wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/config.json                       # config for the distilled model 
wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/tf_model.h5                       # distilled hi-gondi model 
wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/tfb_hi_gondi_28_encoder.tflite    # distilled encoder's offline graph 
wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/tfb_hi_gondi_28_decoder.tflite    # distilled decoder's offline graph
wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/mt5_hi_gondi_decoder.tflite       # quantized decoder's offline graph
wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/mt5_hi_gondi_encoder.tflite       # quantized encoder's offline graph 


# wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/en_hi_pilot_model.zip
# wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/GondiDeploymentModel.zip
# wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/GondiDeploymentTokenizer.zip
wget https://inmtlite.blob.core.windows.net/inmtlite-public-access-models/Hi_Gondi_Deployment_mt5_model.zip