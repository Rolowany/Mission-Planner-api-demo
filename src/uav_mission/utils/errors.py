class MissionError(Exception):
    """Błąd, po którym misja nie może być kontynuowana."""

class MissionFileError(Exception):
    """Plik misji jest niepoprawny. Lista problemów w `errors`."""
    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        details = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(f"Niepoprawna misja:\n{details}")