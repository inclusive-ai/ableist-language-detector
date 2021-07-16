"""Main module for identifying ableist language in job descriptions
to train an unregistered mlflow model."""

import click
import pandas as pd
import mlflow
import mlflow.pyfunc
import json
from ableist_language_detector.detector import find_ableist_language


def load_from_file(filename):
    with open(filename, "r") as jd_file:
        job_description_text = jd_file.read()
    return job_description_text


class MLflowLanguageModel(mlflow.pyfunc.PythonModel):

    def __init__(self, func):
        self.func = func
        self.load_text = load_from_file  # TODO bypass file if needed

    def predict(self, context, model_input):

        input_text = self.load_text(model_input['input_file'][0])
        properties = model_input['properties'][0]
        result = self.func(input_text)

        terms = {}
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
            for ableist_term in result:
                terms[str(ableist_term.start)] = dict()
                if len(properties) > 0:
                    for p in properties:
                        try:
                            terms[str(ableist_term.start)][p] = str(getattr(ableist_term, p))
                        except AttributeError:
                            terms[str(ableist_term.start)][p] = str(getattr(ableist_term.data, p))
        return terms


@click.command()
@click.option(
    "--job_file_json",
    "-j",
    type=str,
    required=True,
    help="json file specifying input text file path.",
)
def main(job_file_json):
    """Extract ableist terms from a job description."""

    model_path = "detector_model"

    # Construct and save the model if one does not exist
    try:
        analyzer = MLflowLanguageModel(find_ableist_language)
        mlflow.pyfunc.save_model(path=model_path, python_model=analyzer)
        print("Generating new model in path {}".format(model_path))

    except:
        print("Using existing model in path {}".format(model_path))
        pass

    # Load the model in `python_function` format
    local_model = mlflow.pyfunc.load_model(model_uri=model_path)

    with open(job_file_json) as f:
        data = json.loads(f.read())
        model_input = pd.json_normalize(data)

    local_output = local_model.predict(model_input)
    print(local_output)


if __name__ == "__main__":
    main()
