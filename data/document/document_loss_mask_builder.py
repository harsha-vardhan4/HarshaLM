from data.document.formatted_document import (
    FormattedDocument,
)

from tokenizer.tokenizer import (
    create_tokenizer,
)


class DocumentLossMaskBuilder:
    """
    Builds GPT training tensors for plain-text documents.

    Every document token contributes
    to the training loss.

    Format:

    <BOS>
    document text
    <EOS>
    """


    def __init__(self):

        self.tokenizer = create_tokenizer()


    def build(
        self,
        document: FormattedDocument,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
    ]:


        #
        # Build sequence
        #

        sequence = [
            self.tokenizer.bos_token_id
        ]


        #
        # DocumentFormatter already adds EOS
        #

        sequence.extend(

            self.tokenizer.encode(
                document.text,
                add_special_tokens=False,
            )

        )


        #
        # GPT next-token prediction
        #

        input_ids = sequence[:-1]

        labels = sequence[1:]


        #
        # Learn every token
        #

        loss_mask = [
            1
        ] * len(input_ids)


        assert (
            len(input_ids)
            ==
            len(labels)
            ==
            len(loss_mask)
        )


        return (
            input_ids,
            labels,
            loss_mask,
        )