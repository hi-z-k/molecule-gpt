"""Character-level tokenizer. Ids are the 4 base specials, extra specials, then sorted characters."""
import json


class CharTokenizer:
    SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"]

    def __init__(self, chars, extra_specials=()):
        extra_specials = [t for t in extra_specials if t not in self.SPECIALS]
        chars = sorted(set(chars) - set(self.SPECIALS) - set(extra_specials))
        self.chars = chars
        self.extra_specials = list(extra_specials)
        self.specials = list(self.SPECIALS) + self.extra_specials   # all special token strings
        self.itos = list(self.specials) + chars              # id -> token string
        self.stoi = {c: i for i, c in enumerate(self.itos)}  # token string -> id
        self.special_ids = set(range(len(self.specials)))    # ids that are not SMILES characters

    @classmethod
    def from_texts(cls, texts, extra_specials=()):
        chars = set()
        for t in texts:
            chars.update(t)
        return cls(sorted(chars), extra_specials)

    pad_id = property(lambda self: 0)
    bos_id = property(lambda self: 1)
    sep_id = property(lambda self: 2)
    eos_id = property(lambda self: 3)

    def id_of(self, token):
        """Id of a special (or character) token string."""
        return self.stoi[token]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s):
        """String -> list of ids. Unknown characters are dropped."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids, skip_special=True):
        """List/tensor of ids -> string."""
        out = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.itos):
                continue
            if skip_special and i in self.special_ids:
                continue
            out.append(self.itos[i])
        return "".join(out)

    def to_dict(self):
        return {"chars": self.chars, "extra_specials": self.extra_specials}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"], d.get("extra_specials", ()))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
