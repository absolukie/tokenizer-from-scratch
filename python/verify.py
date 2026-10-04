"""Verify our from-scratch o200k_base against real tiktoken."""
import random, sys
sys.path.insert(0, 'python')
import tiktoken
from gpt_tokenizer import Tokenizer as BPETokenizer, load_ranks, SPECIAL_TOKENS

ref = tiktoken.get_encoding('o200k_base')
mine = BPETokenizer(load_ranks('o200k_base.tiktoken'), special_tokens=SPECIAL_TOKENS)

random.seed(11)
alphabet = list('abcXYZ 012 \n\t.,!?-<|>e\'"') + ['世', '🌍', 'é', '👨', '\u200d', 'ſ']
corpus = [
    'Hello world!', "It's don't they're I've we'd they'll can't won't", "IT'S DON'T",
    '123 1234 12345 3.14159', '   leading and trailing   ', '\t tabs \n newlines \n\n\n triple',
    'a  b   c    d', 'Hello世界🌍 مرحبا हिन्दी', '👨‍👩‍👧‍👦 family', '🇺🇸🇬🇧',
    'café naïve über', '{"key": "value"}', 'def foo(x):\n    return x * 2',
    '<|endoftext|> split <|endofprompt|> here', 'a' * 200, '!!! ??? ...',
    'email@example.com https://example.com/path?q=1', '\r\n windows \r',
    'The quick brown fox jumps over the lazy dog. ' * 20,
]
for _ in range(3000):
    corpus.append(''.join(random.choice(alphabet) for _ in range(random.randint(0, 80))))

enc_bad = 0
for t in corpus:
    want = ref.encode(t, allowed_special='all')
    got = mine.encode(t, allowed_special='all')
    if want != got:
        enc_bad += 1
        if enc_bad <= 3:
            print('ENCODE DIFF:', repr(t[:70]))
            print('  want:', want[:14])
            print('  got :', got[:14])
print(f'encode: {len(corpus)} texts, {enc_bad} diffs')

# decode: random id sequences must decode identically
dec_bad = 0
for _ in range(3000):
    n = random.randint(1, 12)
    ids = [random.randrange(0, 200019) for _ in range(n)]
    if ref.decode(ids) != mine.decode(ids):
        dec_bad += 1
        if dec_bad <= 3: print('DECODE DIFF:', ids)
print(f'decode: 3000 random id seqs, {dec_bad} diffs')

# roundtrip on tricky strings
rt_bad = sum(1 for t in corpus[:400] if mine.decode(mine.encode(t)) != t)
print(f'roundtrip: {rt_bad} failures on 400 texts')

# special tokens
st = mine.encode('a<|endoftext|>b<|endofprompt|>c')
assert st[1] == 199999 and st[3] == 200018, st
print('special tokens ok:', st)

# vocab stats
print('vocab size:', mine.vocab_size, '| max id:', max(ref.encode('x')))
print('ALL OK' if enc_bad == 0 and dec_bad == 0 and rt_bad == 0 else 'FAILURES PRESENT')
