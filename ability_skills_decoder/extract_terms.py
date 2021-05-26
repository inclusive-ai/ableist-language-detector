"""Module with functions to extract ability vs. skills terms from ONET data."""

from typing import Iterable, Tuple, List, Union
from collections import Counter
from pathlib import Path
import click
import pandas as pd
import spacy


nlp = spacy.load("en_core_web_sm")


def is_verb(token: spacy.tokens.Token) -> bool:
    """Return True if the token is a non-auxiliary verb, else return False.

    Parameters
    ----------
    token : spacy.tokens.Token
        spacy token

    Returns
    -------
    bool
        True if the token is a non-auxiliary verb, else False
    """
    if token.pos_ == "VERB" and token.dep_ not in {"aux", "auxpass", "neg"}:
        return True
    return False


def is_object(token: spacy.tokens.Token) -> bool:
    """Return True if the token is a noun object, else return False.

    Parameters
    ----------
    token : spacy.tokens.Token
        spacy token

    Returns
    -------
    bool
        True if the token is a noun object, else False
    """
    if token.pos_ == "NOUN" and token.dep_ == "dobj":  # direct object dependency tag
        return True
    return False


def get_verbs(
    spacy_doc: spacy.tokens.Doc, return_lemma: bool = True
) -> List[Union[str, spacy.tokens.Token]]:
    """Return a list of verb lemmas within a given document.

    Parameters
    ----------
    spacy_doc : spacy.tokens.Doc
        spaCy document to parse
    return_lemma : bool, optional
        If true, return the string lemmas instead of the spaCy token objects,
        by default True

    Returns
    -------
    List[Union[str, spacy.tokens.Token]]
        A list of tokens or string lemmas
    """
    verbs = [token for token in spacy_doc if is_verb(token)]
    if return_lemma:
        return [token.lemma_ for token in verbs]
    return verbs


def get_objects(
    spacy_doc: spacy.tokens.Doc, return_lemma: bool = True
) -> List[Union[str, spacy.tokens.Token]]:
    """Return a list of noun objects within a given document.

    Parameters
    ----------
    spacy_doc : spacy.tokens.Doc
        spaCy document to parse
    return_lemma : bool, optional
        If true, return the string lemmas instead of the spaCy token objects,
        by default True

    Returns
    -------
    List[Union[str, spacy.tokens.Token]]
        A list of tokens or string lemmas
    """
    noun_objects = [token for token in spacy_doc if is_object(token)]
    if return_lemma:
        return [token.lemma_ for token in noun_objects]
    return noun_objects


def get_nouns(
    spacy_doc: spacy.tokens.Doc, return_lemma: bool = True
) -> List[Union[str, spacy.tokens.Token]]:
    """Return a list of nouns within a given document.

    Parameters
    ----------
    spacy_doc : spacy.tokens.Doc
        spaCy document to parse
    return_lemma : bool, optional
        If true, return the string lemmas instead of the spaCy token objects,
        by default True

    Returns
    -------
    List[Union[str, spacy.tokens.Token]]
        A list of tokens or string lemmas
    """
    nouns = [token for token in spacy_doc if token.pos_ == "NOUN"]
    if return_lemma:
        return [token.lemma_ for token in nouns]
    return nouns


def get_verb_phrases(spacy_doc: spacy.tokens.Doc):
    # inspired by: https://github.com/explosion/spaCy/blob/master/spacy/lang/en/syntax_iterators.py
    # for i, token in enumerate(spacy_doc):
    #     if is_verb(token):
    #         # go the right and check if it's an object
    verb_phrases = []
    for token in spacy_doc:
        if token.dep_ == "dobj":
            phrase = [token.head, [child for child in token.children], token]
            verb_phrases.append(phrase)
    return verb_phrases


def get_abilities(df: pd.DataFrame) -> pd.DataFrame:
    """Given the base content reference dataframe, return the rows that reference
    specific ability descriptions only.

    Parameters
    ----------
    df : pd.DataFrame
        Content model reference dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe containing specific ablities and their descriptions
    """
    # section 1.A.*; abilities are element IDs with 9 elements
    # (fewer than 9 = subcategories)
    abilities_df = df[
        (df["Element ID"].str.startswith("1.A")) & (df["Element ID"].str.len() == 9)
    ]
    return abilities_df


def get_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Given the base content reference dataframe, return the rows that reference
    specific skill descriptions only.

    Parameters
    ----------
    df : pd.DataFrame
        Content model reference dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe containing specific skills and their descriptions
    """
    # either section 2.A or 2.B and element ID = 7 characters long
    skills_df = df[
        (
            (df["Element ID"].str.startswith("2.A"))
            | (df["Element ID"].str.startswith("2.B"))
        )
        & (df["Element ID"].str.len() == 7)
    ]
    return skills_df


def get_representative_terms(
    abilities_corpus: Iterable[str], skills_corpus: Iterable[str]
) -> Tuple[list, list]:
    """Return representative terms in order of importance from the abilities and skills
    corpora.


    Parameters
    ----------
    abilities_corpus : Iterable[str]
        Iterable containing the ability descriptions
    skills_corpus : Iterable[str]
        Iterable containing the skills descriptions

    Returns
    -------
    Tuple[list, list]
        The first element is a list of abilities terms and the second element is a list
        of skills terms
    """
    # Intialize empty lists to store all the verbs extracted from the docs
    abilities_verbs = []
    skills_verbs = []

    # For each description, get the verbs and append them to the master list
    # TODO: Could refine by only retrieving verbs that occur at the start of the
    # description, i.e. only capture the main verb used in the skill/ability
    for doc in nlp.pipe(abilities_corpus):
        abilities_verbs.extend(get_verbs(doc))
    for doc in nlp.pipe(skills_corpus):
        skills_verbs.extend(get_verbs(doc))

    # Get counts for each verb; will be useful for ranking later
    abilities_verbs_counter = Counter(abilities_verbs)
    skills_verbs_counter = Counter(skills_verbs)

    # Compute the set difference and sort by term frequency
    # TODO: Could implement something more sophisticated/closer to TF-IDF that looks at
    # how often a term occurs in abilities vs. skills--impact would be to expand the
    # term list to include terms that occurred in both, but occurred much more
    # frequently in one than another
    unique_abilities_verbs = sorted(
        list(set(abilities_verbs).difference(skills_verbs)),
        key=lambda x: -abilities_verbs_counter[x],
    )
    unique_skills_verbs = sorted(
        list(set(skills_verbs).difference(abilities_verbs)),
        key=lambda x: -skills_verbs_counter[x],
    )
    return unique_abilities_verbs, unique_skills_verbs


def get_objects_corpus(corpus: Iterable[str]) -> list:
    """Return a list of all noun objects in a given corpus.

    Parameters
    ----------
    corpus : Iterable[str]
        Iterable containing individual documents (in this case, skill/ability
        descriptions)

    Returns
    -------
    list
        List of unique noun objects
    """
    noun_objects = []
    for doc in nlp.pipe(corpus):
        noun_objects.extend(get_objects(doc))
    return list(set(noun_objects))


def get_nouns_corpus(corpus: Iterable[str]) -> list:
    """Return a list of all nouns in a given corpus.

    Parameters
    ----------
    corpus : Iterable[str]
        Iterable containing individual documents (in this case, skill/ability
        descriptions)

    Returns
    -------
    list
        List of unique nouns
    """
    nouns = []
    for doc in nlp.pipe(corpus):
        nouns.extend(get_nouns(doc))
    return list(set(nouns))


@click.command()
@click.option(
    "--data_path",
    "-d",
    type=str,
    required=True,
    help=(
        "Local path to raw O*Net Content Model Reference document. Download from: "
        "https://www.onetcenter.org/dictionary/25.2/text/content_model_reference.html"
    ),
)
@click.option(
    "--output_dir",
    "-o",
    type=str,
    required=True,
    help="Path to local directory to save skills and abilities terms lists.",
)
def main(data_path, output_dir):
    """Extract representative terms for abilities and skills."""
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True, exist_ok=True
    )  # Create the subdir(s) if they don't already exist

    df = pd.read_csv(data_path, delimiter="\t")
    abilities_df = get_abilities(df)
    skills_df = get_skills(df)

    unique_abilities_verbs, unique_skills_verbs = get_representative_terms(
        abilities_df.Description, skills_df.Description
    )

    # goal is to retrieve nouns that reference physical (e.g. body parts) or
    # sensory/cognitive abilities that may be ableist. These would all be verb ojects
    # in usage (e.g. move your hand), but looking only for objects was returning
    # limited results, so we'll also look for nouns and then manually curate them later.
    noun_objects = get_objects_corpus(abilities_df.Description)
    nouns = get_nouns_corpus(abilities_df.Description)

    with open(output_path / "abilities_verbs.txt", "w") as abilities_out:
        abilities_out.writelines([f"{v}\n" for v in unique_abilities_verbs])

    with open(output_path / "skills_verbs.txt", "w") as skills_out:
        skills_out.writelines([f"{v}\n" for v in unique_skills_verbs])

    with open(output_path / "abilities_noun_objects.txt", "w") as abilities_objects_out:
        abilities_objects_out.writelines([f"{v}\n" for v in noun_objects])

    with open(output_path / "abilities_nouns.txt", "w") as abilities_nouns_out:
        abilities_nouns_out.writelines([f"{v}\n" for v in nouns])


if __name__ == "__main__":
    main()
