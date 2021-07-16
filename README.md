# [wip] ableist-language-detector
Tool to identify ableist language in job descriptions.

**What is ableist language?**

> Ableist language is language that is offensive to people with disability. It can also refer to language that is derogatory, abusive or negative about disability. Ableism is the systemic exclusion and oppression of people with disability, often expressed and reinforced through language. [[source]](https://pwd.org.au/resources/disability-info/language-guide/ableist-language/)

**Why is this tool important?**

Ableist language in job descriptions can cause people with disabilities to feel excluded from jobs that they are qualified for. This typically occurs when a description references [*abilities*](https://www.onetonline.org/find/descriptor/browse/Abilities/) or enduring attributes of an individual that are unnecessary for the job or for which [accommodations](https://askjan.org/) can be proactively offered instead of focusing on developed [*skills*](https://www.onetonline.org/skills/) that can be acquired to succeed in the role. By identifying ableist language and suggesting alternatives, this tool will support more inclusive hiring practices.

## Installation

Clone the repo and install the package in edit mode (preferably in a virtual environment).
```
git clone git@github.com:inclusive-ai/ableist-language-detector.git
cd ableist-language-detector
pip install -e .
```

Download spaCy dependencies.
```
python -m spacy download en_core_web_sm
```

**Developer Installation**

If you plan on contributing to the repo, complete these additional steps:

Install the dev requirements.

```
pip install -r requirements_dev.txt
```

## Features

* [`extract_onet_terms.py`](ableist_language_detector/extract_terms.py): Extract representative terms for abilities and skills from O*Net data. Used as a source for our ableist lexicon.
* [`detector.py`](ableist_language_detector/detector.py): Main module that identifies ableist language in a job description.

## Basic Usage

To identify ableist language in a job description, pass a `.txt` file containing the job description text to the `detector.py` script:

```
python detector.py -j /path/to/job_description.txt
```

The script will print out any ableist language that was detected in the job description, along with the location of the language (index position in the text), the root form of the terms, suggested alternative verbs, and an example of how to use the alternative phrasing.

The main functionality is also available as a function via `detector.find_ableist_language()`. This function returns a collection of `AbleistLanguageMatch` objects, which contain the same information listed above as attributes.

Example usage:

```python
>>> import spacy
>>> from ableist_language_detector import detector

>>> sample_job_description = """
    requirements
    - must be able to move your hands repeatedly
    - type on a computer
    - comfortable with lifting heavy boxes
    - excellent communication skills
    - move your wrists in circles and bend your arms
"""
>>> ableist_language = detector.find_ableist_language(sample_job_description)
>>> print(ableist_language)
[lifting, bend, move your hands, move your wrists]

# Accessing attributes
def print_results(result):
    """Convenience function to print attributes."""
    print(f"Found {len(result)} instances of ableist language.\n")
    if len(result) > 0:
        for i, ableist_term in enumerate(result):
            print(
                f"Match #{i+1}\n"
                f"PHRASE: {ableist_term} | LEMMA: {ableist_term.lemma} | "
                f"POSITION: {ableist_term.start}:{ableist_term.end} | "
                f"ALTERNATIVES: {ableist_term.data.alternative_verbs} | "
                f"EXAMPLE: {ableist_term.data.example}\n"
            )
>>> print_results(ableist_language)
Found 4 instances of ableist language.

Match #1
PHRASE: lifting | LEMMA: lift | POSITION: 22:23 | ALTERNATIVES: ['move', 'install', 'operate', 'manage', 'put', 'place', 'transfer', 'transport'] | EXAMPLE: Transport boxes from shipping dock to truck

Match #2
PHRASE: bend | LEMMA: bend | POSITION: 38:39 | ALTERNATIVES: ['lower oneself', 'drop', 'move to', 'turn'] | EXAMPLE: Install new ethernet cables under floor rugs

Match #3
PHRASE: move your hands | LEMMA: move your hand | POSITION: 8:11 | ALTERNATIVES: ['observe', 'operate', 'transport', 'transfer', 'activate'] | EXAMPLE: Operates a machine using a lever

Match #4
PHRASE: move your wrists | LEMMA: move your wrist | POSITION: 32:35 | ALTERNATIVES: ['observe', 'operate', 'transport', 'transfer', 'activate'] | EXAMPLE: Operates a machine using a lever
```

## Basic usage with mlflow

Building an mlflow model will allow for usage with a REST API when a server is running.

### Example using the Python API

Pass a `.json` file specifying the job description file and requested parameters to the `detector_mlflow_local.py` script:

```
python detector_mlflow_local.py -j /path/to/job_info.json
```

Example via command line for the included data in `sample_data`:

```python
from ableist_language_detector import detector_mlflow_local as dml
from ableist_language_detector import detector
import json
import mlflow
import pandas as pd

job_file_json = 'sample_data/sample.json'

model_path = 'detector_model'

# Construct and save the model if one does not exist
try:
    analyzer = dml.MLflowLanguageModel(detector.find_ableist_language)
    mlflow.pyfunc.save_model(path=model_path, python_model=analyzer)
    print("Generating new model in path {}".format(model_path))

except:
    print("Using existing model in path {}".format(model_path))
    pass

# Load model
local_model = mlflow.pyfunc.load_model(model_uri=model_path)

# Analyze job description
with open(job_file_json) as f:
      data = json.loads(f.read())
      model_input = pd.json_normalize(data)

local_output = local_model.predict(model_input)

```
`local_output` is a nested dict with keys of start position and values of dicts
for the properties listed in the input `.json` file.

### Example using a REST API

Once a model is saved in the defined path, you can also utilize the REST API with the included shell scripts (after they are made executable).

Serve an input model to port 1234:

```
./serveModel.sh detector_model
```

Once the model is served, in a separate shell run the predict script defining the input json file specify in the job description file to analyze and the requested properties, for example:

```
./predictAPI.sh sample_data/sample.json
```
or
```
./predictAPI.sh sample_data/usa-jobs-astronomer.json
```

which yields:

```
{"558": {"lemma": "carry", "text": "carry", "start": "558", "end": "559", "alternative_verbs": "['move', 'install', 'operate', 'manage', 'put', 'place', 'transfer', 'transport']", "example": "Transport boxes from shipping dock to truck"}, "1107": {"lemma": "lift", "text": "lift", "start": "1107", "end": "1108", "alternative_verbs": "['move', 'install', 'operate', 'manage', 'put', 'place', 'transfer', 'transport']", "example": "Transport boxes from shipping dock to truck"}, "1631": {"lemma": "see", "text": "see", "start": "1631", "end": "1632", "alternative_verbs": "['assess', 'comprehend', 'discover', 'distinguish', 'detect', 'evaluate', 'find', 'identify', 'interpret', 'observe', 'recognize', 'understand']", "example": "Observe any cars illegally parked in the loading zone"}, "1695": {"lemma": "see", "text": "see", "start": "1695", "end": "1696", "alternative_verbs": "['assess', 'comprehend', 'discover', 'distinguish', 'detect', 'evaluate', 'find', 'identify', 'interpret', 'observe', 'recognize', 'understand']", "example": "Observe any cars illegally parked in the loading zone"}}
```
