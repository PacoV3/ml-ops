# ML-Ops

POC involving 3 datasets from banking, shopping and power consumption converted into ML models redy to be deployed.

## Usage

### For local Development

The following command creates a docker container that runs PySpark on Jupyter Notebooks to be able to connect to localhost:8888 with the token it provides on the terminal.

```sh
docker run --rm -it -p 8888:8888 -v ${PWD}:/work -w /work jupyter/pyspark-notebook
```
