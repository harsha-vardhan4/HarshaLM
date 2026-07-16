from data.document.formatted_document import FormattedDocument
from data.training_corpus import TrainingCorpus
from data.validation.corpus_validator import CorpusValidator


def test_duplicate_document():

    corpus = TrainingCorpus(
        dataset_name="test"
    )

    corpus.documents.append(
        FormattedDocument("Hello")
    )

    corpus.documents.append(
        FormattedDocument("Hello")
    )

    validator = CorpusValidator()

    validated, report = validator.validate(corpus)

    assert len(validated.documents) == 1

    assert report.duplicate_documents == 1