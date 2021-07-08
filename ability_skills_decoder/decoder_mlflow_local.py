"""Module for identifying ableist language in job descriptions
to train an unregistered mlflow model."""

import click
import pandas as pd
import mlflow
import mlflow.pyfunc
import json
from ability_skills_decoder.decoder import find_ableist_language

import spacy


def load_from_file(filename):
    with open(filename, "r") as jd_file:
        job_description_text = jd_file.read()
    return job_description_text


class MLflowLanguageModel(mlflow.pyfunc.PythonModel):

    def __init__(self, func):
        self.func = func
        self.load_text = load_from_file  # TODO bypass file

    def predict(self, context, model_input):

        input_text = self.load_text(model_input['input'][0])
        properties = model_input['properties'][0]
        results = self.func(input_text)

        terms = {}
        for ableist_term in results:
            if isinstance(ableist_term, spacy.tokens.Span):
                print(
                    f"PHRASE: {ableist_term} | LEMMA: {ableist_term.lemma_} | "
                    f"POSITION: {ableist_term.start}:{ableist_term.end}"
                )
                terms[f"{ableist_term.start}:{ableist_term.end}"] = dict()
                terms[f"{ableist_term.start}:{ableist_term.end}"]['ableist_term'] = str(ableist_term)
                for at in properties:
                    terms[f"{ableist_term.start}:{ableist_term.end}"][at] = getattr(ableist_term, at)
            else:
                print(
                    f"PHRASE: {ableist_term} | LEMMA: {ableist_term.lemma_} | "
                    f"POSITION: {ableist_term.i}"
                )
                terms[str(ableist_term.i)] = dict()
                terms[str(ableist_term.i)]['ableist_term'] = str(ableist_term)
                for at in properties:
                    terms[str(ableist_term.i)][at] = getattr(ableist_term, at)
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

    model_path = "analyzer_model"

    # Construct and save the model
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
