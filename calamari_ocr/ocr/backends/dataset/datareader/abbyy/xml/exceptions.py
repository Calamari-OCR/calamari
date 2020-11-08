class WrongFileStructureException(Exception):
    def __init__(self, message: str):
        super(WrongFileStructureException, self).__init__(message)


class XMLParseError(Exception):
    def __init__(self, message: str):
        super(XMLParseError, self).__init__(message)

