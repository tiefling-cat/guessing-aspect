# guessing-aspect
Machine learning aspect detection for Russian verbs.

## About
The work is done as a part of VerbNet project developed at *Higher School of Economics, Moscow, Philology department, master's programme [Computational Linguistics](https://www.hse.ru/en/ma/ling/)* For further information on the project, visit http://web-corpora.net/wsgi3/ru-verbs/

### What's that?
Here is the tool for predicting aspect of Russian verbs. Sure, it's entirely okay to map verb's aspect by hand as Russian native speaker can determine that in a matter of milliseconds, but when you have a large list of verbs mined from somewhere on the Internet, the task becomes a bit frustrating. Here, you just upload a file with your verbs, and get the aspects for them with machine learning classifier trained on nearly 6800 verbs. Several classifiers were tested for that purpose with LinearSVC with l1 penalty proving to be the best in terms of accuracy.

![Benchmarks](http://web-corpora.net/wsgi3/ru-verbs/static/pictures/benchmarks-white.png)

This one here is a command line tool, but you can also use it online at http://web-corpora.net/wsgi3/ru-verbs/aspect

### How's it done?
The whole thing is done the following way. Here we have two classes, perfect and imperfect. For each verb, continuous n-grams of lengts 2 to 4 are produced from its letters. The list of n-grams for each verb is then treated as a small text. Then the problem of classifying texts into two classes is solved, with usual tfidf feature extraction. Since n-grams are a good approximation of word's morphemes, this works as if we were trying to guess aspect by morphemes of the verb. And trust me, I've first done it with morpheme analyser, and it performed no better than n-grams.

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

## Related repos
You can also:
* [Parse yourself some synsets from synonym dictionary](https://github.com/tiefling-cat/bparser)
* [Explore a proto semantic graph of verbs made from word2vec model](https://github.com/tiefling-cat/verb2graph)
