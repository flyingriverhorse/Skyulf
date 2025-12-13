class DataIngestionException(Exception):
    """Base exception for data ingestion errors."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
