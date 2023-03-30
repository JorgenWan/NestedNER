

class Node:
    """
        A node class supporting Hash().
        It can be extended to support more node features.
    """

    def __init__(self, tokens, type):
        self._tokens = tokens
        self._type = type

        self.max_tokens_len = 100
        self.max_type_len = 100

    def __hash__(self):
        tokens_str = " ".join(self._tokens)
        padded_tokens = tokens_str + (self.max_tokens_len - len(tokens_str)) * " "
        padded_type = self._type + (self.max_type_len - len(self._type)) * " "
        return hash(padded_tokens + padded_type)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        else:
            return False

    @property
    def tokens(self):
        return self._tokens

    @property
    def type(self):
        return self._type

if __name__ == "__main__":
    a = set()
    a.add(Node(["a", "b"], "art"))
    a.add(Node(["a", "b"], "art"))
    print(a)