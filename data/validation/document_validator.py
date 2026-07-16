from data.validation.base_validator import (
    BaseValidator,
)

from data.document.formatted_document import (
    FormattedDocument,
)


class DocumentValidator(BaseValidator):
    """
    Validates formatted documents.
    """

    def validate(
        self,
        document: FormattedDocument,
    ) -> bool:

        return bool(
            document.text.strip()
        )