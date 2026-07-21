# Simulator Validation

MARS is built using a rebuilt C++ version of [CQSIM](https://github.com/SPEAR-UIC/CQSim), originally written in Python. To validate the C++ version, we tested the output of the C++ against the known and previously validated Python version.

The output of the Python version can be found under [cqsim-python/data/Results](cqsim-python/data/Results). Here we have extracted the scheduling events in csv files after repalying the logs under various heursitc scehduling polices. Similarly, for the C++ version the [results](results/) folder contains the csv outputs for the C++ version.

Here we provide the steps to recreate the output files and run the tests using the provided scripts.

# Recreating CQSIM Python outputs

# Recreating CQSIM C++ outputs

