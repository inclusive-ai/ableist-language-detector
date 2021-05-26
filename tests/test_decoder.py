#!/usr/bin/env python

"""Tests for decoder functions."""

import spacy
from ability_skills_decoder import decoder

nlp = spacy.load("en_core_web_sm")


def test_match_dependent_ableist_verbs():
    doc = nlp("must be able to move your hands repeatedly")
    ableist_verbs_object_dependent = {"move"}
    ableist_objects = {"hand", "foot"}
    str_result = decoder.match_dependent_ableist_verbs(
        doc, ableist_verbs_object_dependent, ableist_objects
    )[0].text
    assert str_result == "move your hands"