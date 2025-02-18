# A Modern Approach to training ERA5-based Machine Learning Weather Prediction Systems
In this paper, we propose an approach to efficiently train machine learning models for weather prediction. We identify three core issues with ERA5-based models: (1) small dataset size, (2) lack of focus on rare events such as hurricanes and tropical storms, and (3) inefficient computational scaling from coarse to fine resolutions. To address these challenges, we propose three key additions to the training pipeline: adaptive-time dynamics simulation, importance sampling, and multi-resolution knowledge transfer. Our goal is to maximize both data and computational efficiency, ensuring less training and retraining.

#### Dataset 
Download the original datasets from WeatherBench 2 (at multiple resolutions):
```
bash scripts/download_dataset.sh ERA ERA
```

Then to process the datasets:
```
python scripts/preprocess_weatherbench_data.py -i ERA5/5.625deg -o ERA5/5.625deg_processed
```
```
python scripts/preprocess_weatherbench_data.py -i ERA5/1.5deg -o ERA5/1.5deg_processed
```

Then to remove the unprocessed data:
```
bash scripts/cleanup.py ERA/5.625deg ERA/1.5deg
```
     
#### Acknowledgements

This project draws significant inspiration and makes entenstive use of the data pipeline, from the paper [PARADIS - yet to be released].
