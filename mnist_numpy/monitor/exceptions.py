class AbortTraining(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message + (" " if message else "") + "Aborting training.")
