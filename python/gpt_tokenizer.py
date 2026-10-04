"""
o200k_base (the GPT-4o tokenizer) implemented from scratch.

No tiktoken dependency. Loads the public merge ranks, applies the exact
pre-tokenizer regex, and runs byte-level BPE merges.

Layout of the algorithm:
  1. Split out special tokens (<|endoftext|>, <|endofprompt|>) if allowed.
  2. Split each remaining span with the pre-tokenizer regex (finditer).
  3. UTF-8 encode each regex piece -> bytes.
  4. Byte-pair encode: repeatedly merge the adjacent pair with the lowest
     merge rank until no mergeable pair remains.
  5. Map every merged byte chunk to its token id via the ranks table.
"""

import base64

try:
    import regex as re  # supports possessive quantifiers + \p{...}
except ImportError:  # pragma: no cover
    raise ImportError("pip install regex")

# Pre-tokenizer for o200k_base. A subtle but load-bearing detail: OpenAI
# publishes this pattern with *possessive* quantifiers (?+, *+, ++), but
# tiktoken's core is Rust, and the Rust `regex` crate has no possessive
# quantifiers -- it parses `a++` as nested `(a+)+`, i.e. plain greedy.
# Verified empirically (Oct 2026): real tiktoken output matches the GREEDY
# reading wherever possessive vs greedy differ (e.g. 'a  b' splits as
# ['a', ' ', ' b'], never ['a', '  ', 'b']). So the pattern below uses
# greedy quantifiers, exactly as tiktoken's engine compiles it.
PAT_STR = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}


def load_ranks(path):
    """Parse a .tiktoken file -> {token_bytes: rank}."""
    ranks = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64, rank = line.split()
            ranks[base64.b64decode(b64)] = int(rank)
    return ranks


class Tokenizer:
    def __init__(self, ranks, special_tokens=None, pat_str=PAT_STR):
        self.encoder = dict(ranks)  # bytes -> id
        self.decoder = {}
        for tok, idx in ranks.items():
            self.decoder[idx] = tok
        self.special_tokens = dict(special_tokens or {})
        for text, idx in self.special_tokens.items():
            self.decoder[idx] = text.encode("utf-8")
        self.pat = re.compile(pat_str)
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pat = re.compile("|".join(re.escape(s) for s in ordered))
        else:
            self.special_pat = None

    # -- core BPE ---------------------------------------------------------
    def _byte_pair_encode(self, piece: bytes):
        """Merge the lowest-ranked adjacent pair until none is mergeable."""
        if len(piece) <= 1:
            return [self.encoder[piece]]
        parts = [(i, i + 1) for i in range(len(piece))]
        while True:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                merged = piece[parts[i][0] : parts[i + 1][1]]
                rank = self.encoder.get(merged)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_rank is None:
                break
            s = parts[best_i][0]
            e = parts[best_i + 1][1]
            parts = parts[:best_i] + [(s, e)] + parts[best_i + 2 :]
        return [self.encoder[piece[s:e]] for s, e in parts]

    def _encode_ordinary(self, text: str):
        ids = []
        for m in self.pat.finditer(text):
            piece = m.group(0).encode("utf-8")
            ids.extend(self._byte_pair_encode(piece))
        return ids

    # -- public API --------------------------------------------------------
    def encode(self, text, allowed_special="all", disallowed_special="all"):
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif allowed_special == "none":
            allowed = set()
        else:
            allowed = set(allowed_special)
        if disallowed_special == "all":
            disallowed = set(self.special_tokens) - allowed
        elif disallowed_special == "none":
            disallowed = set()
        else:
            disallowed = set(disallowed_special) - allowed

        if not self.special_pat or (not allowed and not disallowed):
            return self._encode_ordinary(text)

        ids = []
        pos = 0
        for m in self.special_pat.finditer(text):
            chunk = text[pos : m.start()]
            if chunk:
                ids.extend(self._encode_ordinary(chunk))
            tok = m.group(0)
            if tok in disallowed:
                raise ValueError(f"disallowed special token: {tok!r}")
            if tok in allowed:
                ids.append(self.special_tokens[tok])
            else:  # not allowed and not disallowed -> treat as ordinary text
                ids.extend(self._encode_ordinary(tok))
            pos = m.end()
        tail = text[pos:]
        if tail:
            ids.extend(self._encode_ordinary(tail))
        return ids

    def decode(self, ids):
        return b"".join(self.decoder[i] for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return max(self.decoder) + 1


def load_o200k(path="o200k_base.tiktoken"):
    return Tokenizer(load_ranks(path), SPECIAL_TOKENS)


if __name__ == "__main__":
    import sys

    tok = load_o200k(sys.argv[1] if len(sys.argv) > 1 else "o200k_base.tiktoken")
    demo = "Hello world! It's a beautiful day. <|endoftext|> 12345"
    ids = tok.encode(demo)
    print(ids)
    print(tok.decode(ids))
    print("roundtrip ok:", tok.decode(ids) == demo)
