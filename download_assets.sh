#!bin/bash
# If this fails, install/updgrade huggingface_hub, i.e. pip3 install -U huggingface_hub
mkdir assets && hf sync hf://buckets/e-zorzi/ithor_scenes/ assets/
echo ">>>> If this fails, install/upgrade huggingface_hub, i.e. pip3 install -U huggingface_hub and re-run it"
