from dataLoader import DataLoader
from clusterAnalysis import clusterAnalysis
from TS_Analysis import TS_Analysis
from forecastModel import forecastModel
import os
import warnings


def main(cluster=False):
    loader = DataLoader()
    resampledData = loader.getResampledData()
    if cluster:
        clusterAnalysis(loader.getData(), explore=False)
    TS_Analysis(resampledData)
    model = forecastModel(resampledData)


if __name__== "__main__":
    os.environ['KMP_WARNINGS'] = 'off'        # openmp warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorflow deprications
    warnings.filterwarnings("ignore")
    main()
