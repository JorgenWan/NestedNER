class Token:
    def __init__(self, tok_id, index, span_start, span_end, phrase):
        self._tok_id = tok_id  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (inclusive)
        self._phrase = phrase

    @property
    def num_bpe(self):
        return self._span_end - self._span_start + 1

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start # BPE-level position

    @property
    def span_end(self):
        return self._span_end # BPE-level position

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tok_id == other._tok_id
        return False

    def __hash__(self):
        return hash(self._tok_id)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase

class TokenSpan:
    """
    For Document and Entity classes to return .tokens
    """
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span(self):
        return self._tokens[0].span_start, self._tokens[-1].span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Entity:
    def __init__(self, ent_id, entity_type_id, tokens, phrase, start, end):
        self._ent_id = ent_id  # ID within the corresponding dataset

        self._entity_type_id = entity_type_id # int
        self._tokens = tokens # OrderedDict(Token)
        self._phrase = phrase  # str

        self._start = start # int
        self._end = end # int

    def as_tuple(self):
        # return self.span_start, self.span_end, self._entity_type_id
        return self._start, self._end, self._entity_type_id

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def entity_type_id(self):
        return self._entity_type_id

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._ent_id == other._ent_id
        return False

    def __hash__(self):
        return hash(self._ent_id)

    def __str__(self):
        return self._phrase


class Document:
    def __init__(
        self, doc_id, tokens, entities, bpe_ids, bpes,
        chars, words
    ):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._entities = entities
        self._bpe_ids = bpe_ids # BPE ids with [CLS] and [SEP]
        self._bpes = bpes # BPE strings with [CLS] and [SEP]

        self._char_ids = [] # char ids without [CLS] and [SEP]
        self._chars = chars # strs without [CLS] and [SEP]

        self._word_ids = [] # word ids without [CLS] and [SEP]
        self._words = words # strs without [CLS] and [SEP]

        self._text = " ".join(words)


    def set_char_ids(self, char2idx):
        n_unk_ids = 0
        n_ids = 0
        char_ids = []
        for cc in self._chars:
            each_char_ids = []
            for c in cc:
                if c in char2idx.keys():
                    each_char_ids.append(char2idx[c])
                else:
                    each_char_ids.append(char2idx["<unk>"])
                    n_unk_ids += 1
                n_ids += 1
            char_ids.append(each_char_ids)
        self._char_ids = char_ids
        return n_ids, n_unk_ids

    def set_word_ids(self, word2idx):
        n_unk_ids = 0
        n_ids = 0
        word_ids = []
        for w in self._words:
            if w in word2idx.keys():
                word_ids.append(word2idx[w])
            else:
                word_ids.append(word2idx["<unk>"])
                n_unk_ids += 1
            n_ids += 1
        self._word_ids = word_ids
        return n_ids, n_unk_ids

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def entities(self):
        return self._entities

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def bpe_ids(self):
        return self._bpe_ids

    @property
    def bpes(self):
        return self._bpes

    @property
    def chars(self):
        return self._chars

    @property
    def char_ids(self):
        return self._char_ids

    @property
    def words(self):
        return self._words

    @property
    def word_ids(self):
        return self._word_ids

    @property
    def text(self):
        return self._text

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)
