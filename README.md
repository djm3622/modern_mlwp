# modern_mlwp


#### Dataset 
Download the original datasets from WeatherBench 2 (at multiple resolutions):
```
bash scripts/download_dataset.sh ERA ERA
```

Then to process the datasets:
```
python scripts/preprocess_weatherbench_data.py -i ERA5/5.625deg -o ERA5/5.625deg_process
```
```
python scripts/preprocess_weatherbench_data.py -i ERA5/1.5deg -o ERA5/1.5deg_process
```

Then to remove the unprocessed data:
```
bash scripts/cleanup.py ERA/5.625deg ERA/1.5deg
```
     
#### Acknowledgements

This project draws significant inspiration and makes entenstive use of the data pipeline, from the paper [PARADIS - yet to be released].
