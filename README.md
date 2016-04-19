# guessing-aspect
Machine learning aspect detection for Russian verbs.

## Requirements
* python 3.4+
* numpy
* sklearn
* matplotlib

## Usage
```
Usage: python3 aspect_tools.py [OPTIONS] [INPUT_FILE]

Options:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output=OUTPUT_FILE
                        Output results to OUTPUT_FILE. Default is
                        INPUT_FILE_classified_by_CLASSIFIER_NAME.csv
  -c CLASSIFIER_NAME, --classifier=CLASSIFIER_NAME
                        Specify classifier (LinearSVC1 is default)
  -l, --list-classifiers
                        List available classifiers
  -b, --benchmark       Run classifier benchmarks on training data
  -t, --top10           Print ten most discriminative n-grams per class for
                        every classifier
  -s, --silent          Supress output
  -m LOG_FILE, --benchmark-log=LOG_FILE
                        Output all benchmark info to LOG_FILE
  -p GRAPH_OUTPUT_FILE, --plot-output=GRAPH_OUTPUT_FILE
                        Save benchmark diagram to GRAPH_OUTPUT_FILE. File
                        extension must be either png or pdf
```
