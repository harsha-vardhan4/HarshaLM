from pathlib import Path


from data.training_corpus import (
    TrainingCorpus,
)


from data.converters.documents.plain_text_converter import (
    PlainTextConverter,
)


from data.converters.daily_dialog_converter import (
    DailyDialogConverter,
)



class DatasetLoader:
    """
    Finds dataset files and routes them
    to the correct converter.

    Supported:

    .txt -> PlainTextConverter
    .csv -> DailyDialogConverter
    """



    def __init__(self):

        self.text_converter = (
            PlainTextConverter()
        )

        self.csv_converter = (
            DailyDialogConverter()
        )



    def load(
        self,
        path: str,
    ) -> TrainingCorpus:


        path = Path(path)


        if not path.exists():

            raise FileNotFoundError(
                path
            )


        corpus = TrainingCorpus(
            dataset_name=path.name
        )



        files = []


        if path.is_file():

            files.append(
                path
            )


        elif path.is_dir():

            files.extend(
                sorted(
                    path.rglob("*")
                )
            )



        for file in files:


            if not file.is_file():

                continue



            if file.suffix == ".txt":


                partial = self.text_converter.convert(
                    str(file)
                )



            elif file.suffix == ".csv":


                partial = self.csv_converter.convert(
                    str(file)
                )



            else:

                continue



            corpus.documents.extend(
                partial.documents
            )


            corpus.conversations.extend(
                partial.conversations
            )



        return corpus