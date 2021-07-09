"""Module for identifying ableist language in job descriptions
to train and register an mlflow model."""

import click
import pandas as pd
import mlflow
import mlflow.pyfunc
import time
import json
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

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
                terms[str(ableist_term.start)] = dict()
                terms[str(ableist_term.start)]['ableist_term'] = str(ableist_term)
                terms[str(ableist_term.start)]['end'] = str(ableist_term.end)
                for at in properties:
                    terms[str(ableist_term.start)][at] = getattr(ableist_term, at)
            else:
                print(
                    f"PHRASE: {ableist_term} | LEMMA: {ableist_term.lemma_} | "
                    f"POSITION: {ableist_term.i}"
                )
                terms[str(ableist_term.i)] = dict()
                terms[str(ableist_term.i)]['ableist_term'] = str(ableist_term)
                terms[str(ableist_term.i)]['end'] = str(ableist_term.i)
                for at in properties:
                    terms[str(ableist_term.i)][at] = getattr(ableist_term, at)
        return terms


def wait_model_transition(model_name, model_version, stage):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name,
                                                         version=model_version,
                                                         )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print("Model status: %s" % ModelVersionStatus.to_string(status))
        if status == ModelVersionStatus.READY:
            client.transition_model_version_stage(
              name=model_name,
              version=model_version,
              stage=stage,
            )
            break
        time.sleep(1)


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

    client = MlflowClient()

    with mlflow.start_run() as run:
        run_num = run.info.run_id
        model_path = "analyzer_model"
        stage = "Staging"

        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_num,
            artifact_path=model_path)

        # Construct and save the model
        analyzer = MLflowLanguageModel(find_ableist_language)

        # Register model
        mlflow.pyfunc.log_model(model_path, python_model=analyzer)
        mlflow.register_model(model_uri=model_uri,
                              name=model_path)

        model_version_infos = client.search_model_versions("name = '%s'" % model_path)
        new_model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])

        # Add a description
        client.update_model_version(
          name=model_path,
          version=new_model_version,
          description="Ableist job description analyzer"
        )

        try:
            wait_model_transition(model_path, new_model_version - 1, "None")
        except Exception as e:
            print(e)
            pass

        wait_model_transition(model_path, new_model_version, stage)

        # Load the model in `python_function` format
        loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_path}/{stage}")

        with open(job_file_json) as f:
            data = json.loads(f.read())
            model_input = pd.json_normalize(data)

        model_output = loaded_model.predict(model_input)
        print(model_output)


if __name__ == "__main__":
    main()
