def peek_iter(collection):
    return PeekIterator(collection)


class PeekIterator(object):
    """Iterator that allows us to peek at the next item"""

    def __init__(self, collection):
        self.iter = iter(collection)
        self.peeked = next(self.iter, None)

    def __iter__(self):
        return self

    def __next__(self):
        peeked = self.peeked
        self.peeked = next(self.iter, None)
        return peeked

    def peek(self):
        return self.peeked

    def has_next(self):
        return self.peeked
