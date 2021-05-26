"""Main module for identifying ableist terms in job descriptions."""

import click
import spacy
from ability_skills_decoder import extract_terms
from ability_skills_decoder.ableist_word_list import (
    ABLEIST_VERBS,
    ABLEIST_OBJECTS,
    ABLEIST_VERBS_OBJECT_DEPENDENT,
)

nlp = spacy.load("en_core_web_sm")


def match_dependent_ableist_verbs(doc, ableist_verbs_object_dependent, ableist_objects):
    matched_phrases = []
    # reference: https://spacy.io/usage/linguistic-features#navigating
    for token in doc:
        if token.dep_ == "dobj" and token.lemma_ in ableist_objects:
            if (
                extract_terms.is_verb(token.head)
                and token.head.lemma_ in ableist_verbs_object_dependent
            ):
                # some options to return the entire phrase containing the verb + object
                # matched_phrase = [token.head, token]
                # matched_phrase = doc[token.head.i : token.i + 1]  # kind of manual, doesn't consider syntactic descendants
                matched_phrase = doc[
                    token.head.i : token.right_edge.i + 1
                ]  # probably the best one to return the verb + object
                # matched_phrase = doc[
                #     token.head.i : token.head.right_edge.i + 1
                # ]  # most expansive, gets any modifiers of the object that occur after the object, e.g. "move your hands repeatedly"
                matched_phrases.append(matched_phrase)
    return matched_phrases


def find_ableist_terms(job_description_text: str) -> list:
    # Read in jd and convert to spacy doc (could lemmatize to make faster), extract verbs
    job_description_doc = nlp(job_description_text)

    # Match verbs in ableist verb list
    jd_verbs = extract_terms.get_verbs(job_description_doc, return_lemma=False)
    matched_verbs = [verb for verb in jd_verbs if verb.lemma_ in ABLEIST_VERBS]

    # Match verb + object in ableist verb + object list
    jd_verb_phrases = extract_terms.get_verb_phrases(job_description_doc)
    print(jd_verb_phrases)

    # Return the ableist terms and their spans (to enable highlighting later)
    return matched_verbs


@click.command()
@click.option(
    "--job_description_file",
    "-j",
    type=str,
    required=True,
    help="Path to file containing the job description text.",
)
def main(job_description_file):
    """Extract ableist terms from a job description."""
    with open(job_description_file, "r") as jd_file:
        job_description_text = jd_file.read()

    result = find_ableist_terms(job_description_text)
    for ableist_term in result:
        # token position, token, lemma
        print(ableist_term.i, ableist_term, ableist_term.lemma_)


if __name__ == "__main__":
    # main()
    doc = nlp("must be able to move your hands repeatedly")
    # for token in doc:
    # print(
    #     token.text,
    #     token.dep_,
    #     token.head.text,
    #     token.head.pos_,
    #     [child for child in token.children],
    # )
    ableist_verbs_object_dependent = {"move"}
    ableist_objects = {"hand", "foot"}
    res = match_dependent_ableist_verbs(
        doc, ableist_verbs_object_dependent, ableist_objects
    )

    print(res)