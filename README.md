# o200k, from scratch

The GPT-4o tokenizer (`o200k_base`) reimplemented from zero — no `tiktoken` dependency, no ML libraries. **Byte-identical to OpenAI's tiktoken**, verified on thousands of adversarial inputs.

- **`web/tokenizer.js`** — dependency-free JavaScript implementation (powers the playground)
- **`python/gpt_tokenizer.py`** — dependency-free Python implementation (only needs the `regex` module for `\p{...}` classes)
- **`web/index.html`** — interactive playground: live encoder, BPE step animator, train-your-own BPE, vocabulary browser
- **`o200k_base.tiktoken`** — OpenAI's public merge ranks (the vocabulary *data*)
- **`python/verify.py`** — differential test suite against real tiktoken

Live playground: *(URL after deploy)*

## What's implemented

| Piece | Detail |
|---|---|
| Vocabulary | 200,019 ids: 199,998 mergeable ranks + `<\|endoftext\|>` (199999) + `<\|endofprompt\|>` (200018); 19 ids unassigned |
| Pre-tokenizer | The exact 7-branch multilingual regex (words, contractions, numbers ≤3 digits, punctuation, newlines, whitespace) |
| Encoding | Special-token split → regex pieces → UTF-8 bytes → lowest-rank pair merges → ids |
| Decoding | Ids → bytes → UTF-8 (invalid ids raise, like tiktoken) |
| Training | Classic Sennrich BPE: most-frequent pair merges (playground's Train tab) |

## The possessive-quantifier quirk

OpenAI publishes the pre-tokenizer regex with *possessive* quantifiers (`?+`, `*+`, `++`). But tiktoken's core is Rust, and Rust's `regex` crate has no possessive quantifiers — it parses `a++` as nested `(a+)+`, i.e. plain greedy. That changes real output: `'a  b'` tokenizes as `['a', ' ', ' b']` in tiktoken, never `['a', '  ', 'b']`.

This implementation matches tiktoken's **actual** behavior, not the published pattern. A naïve port of the possessive regex (or a JS atomic-group emulation of it) produces *different* tokens — we verified this the hard way.

## Verify it yourself

```bash
python3 -m venv .venv && .venv/bin/pip install tiktoken regex
.venv/bin/python python/verify.py
# encode: 3019 texts, 0 diffs
# decode: 3000 random id seqs, 0 diffs
# roundtrip: 0 failures on 400 texts
# ALL OK
```

The JS implementation is checked the same way (`/tmp/jsverify.mjs` pattern): 3,019 encode vectors + 2,000 decode vectors, 0 diffs.

## Data provenance

The engine is from scratch. The vocabulary *data* — token bytes → merge ranks — is OpenAI's public `o200k_base.tiktoken` file (you can't learn GPT-4o's merges without GPT-4o's training data). The web build ships a compact 1.6 MB `ranks.bin` generated from it (`python` one-liner in the repo history); the original `.tiktoken` file stays canonical.

## License

MIT for the code. The `o200k_base` vocabulary data is OpenAI's, published for use with tiktoken-compatible implementations.
