/* o200k_base (GPT-4o tokenizer) from scratch — no dependencies.
 *
 * Mirrors python/gpt_tokenizer.py exactly:
 *  1. split out special tokens,  2. pre-tokenizer regex (finditer),
 *  3. UTF-8 -> bytes,            4. byte-pair merges by lowest rank,
 *  5. bytes -> token ids.
 *
 * NOTE on the regex: OpenAI publishes the pattern with *possessive*
 * quantifiers (?+, *+, ++), but tiktoken's core is Rust and the Rust
 * `regex` crate has no possessive quantifiers — it parses `a++` as nested
 * `(a+)+`, i.e. plain greedy. Real tiktoken output matches the greedy
 * reading (verified empirically), so the pattern below is the greedy form,
 * exactly as tiktoken's engine compiles it. No possessive/atomic emulation
 * needed — plain JS RegExp with the `u` flag is byte-identical.
 */

const CONTRACTION = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?";

const PAT_STR = [
  "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+" + CONTRACTION,
  "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*" + CONTRACTION,
  "\\p{N}{1,3}",
  " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
  "\\s*[\\r\\n]+",
  "\\s+(?!\\S)",
  "\\s+",
].join('|');

const PRETOKENIZER = new RegExp(PAT_STR, 'gu');

export const SPECIAL_TOKENS = { '<|endoftext|>': 199999, '<|endofprompt|>': 200018 };

/** Parse the compact ranks.bin format -> {binaryString: rank}.
 *  Layout: u32 count, then count u8 lengths, then concatenated token bytes.
 *  (rank == index; the .tiktoken file ships sorted by rank) */
export function parseRanksBin(buffer) {
  const dv = new DataView(buffer);
  const n = dv.getUint32(0, true);
  const lengths = new Uint8Array(buffer, 4, n);
  let dataOff = 4 + n;
  const bytes = new Uint8Array(buffer);
  const ranks = new Map();
  // build binary strings in chunks to avoid O(n^2) concatenation
  let off = dataOff;
  for (let i = 0; i < n; i++) {
    const len = lengths[i];
    let s = '';
    for (let j = 0; j < len; j++) s += String.fromCharCode(bytes[off + j]);
    ranks.set(s, i);
    off += len;
  }
  return ranks;
}

/** Parse a .tiktoken file's text -> {binaryString: rank}. */
export function parseRanks(tiktokenText) {
  const ranks = new Map();
  const lines = tiktokenText.split('\n');
  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    const sp = t.indexOf(' ');
    const b64 = t.slice(0, sp);
    const rank = parseInt(t.slice(sp + 1), 10);
    ranks.set(b64ToBin(b64), rank);
  }
  return ranks;
}

function b64ToBin(b64) {
  const bin = atob(b64);
  // atob gives a binary string already (each char = one byte)
  return bin;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8');
const tick = () => new Promise(r => setTimeout(r, 0));

function utf8ToBin(str) {
  const bytes = encoder.encode(str);
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

function binToBytes(bin) {
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Split text into pre-tokenizer regex pieces (for the playground's piece view). */
export function pretokenize(text) {
  const out = [];
  PRETOKENIZER.lastIndex = 0;
  let m;
  while ((m = PRETOKENIZER.exec(text)) !== null) {
    if (m[0] === '') { PRETOKENIZER.lastIndex++; continue; }
    out.push(m[0]);
  }
  return out;
}

export class BPETokenizer {
  constructor(ranks, specialTokens = SPECIAL_TOKENS) {
    this.ranks = ranks; // Map binaryString -> id
    this.specialTokens = specialTokens;
    this.decoder = new Map();
    for (const [tok, id] of ranks) this.decoder.set(id, tok);
    for (const [text, id] of Object.entries(specialTokens)) {
      this.decoder.set(id, utf8ToBin(text));
    }
    const ordered = Object.keys(specialTokens).sort((a, b) => b.length - a.length);
    this.specialRe = ordered.length
      ? new RegExp(ordered.map(escapeRe).join('|'), 'g')
      : null;
  }

  /** Trace every merge for one binary-string piece: [{parts, pair, rank}]. */
  bytePairTrace(piece /* binary string */) {
    const steps = [];
    if (piece.length <= 1) return steps;
    let parts = [];
    for (let i = 0; i < piece.length; i++) parts.push(piece[i]);
    for (;;) {
      let bestRank = undefined, bestI = -1, bestPair = null;
      for (let i = 0; i < parts.length - 1; i++) {
        const merged = parts[i] + parts[i + 1];
        const r = this.ranks.get(merged);
        if (r !== undefined && (bestRank === undefined || r < bestRank)) {
          bestRank = r; bestI = i; bestPair = [parts[i], parts[i + 1]];
        }
      }
      if (bestRank === undefined) break;
      const done = parts.slice(0, bestI)
        .concat([bestPair[0] + bestPair[1]])
        .concat(parts.slice(bestI + 2));
      steps.push({ parts: parts.slice(), pair: bestPair, rank: bestRank, result: done.slice() });
      parts = done;
    }
    return steps;
  }

  _bytePairEncode(piece /* binary string */) {
    if (piece.length <= 1) return [this.ranks.get(piece)];
    let parts = [];
    for (let i = 0; i < piece.length; i++) parts.push([i, i + 1]);
    for (;;) {
      let bestRank = undefined, bestI = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const merged = piece.slice(parts[i][0], parts[i + 1][1]);
        const r = this.ranks.get(merged);
        if (r !== undefined && (bestRank === undefined || r < bestRank)) {
          bestRank = r; bestI = i;
        }
      }
      if (bestRank === undefined) break;
      const s = parts[bestI][0], e = parts[bestI + 1][1];
      parts = parts.slice(0, bestI).concat([[s, e]], parts.slice(bestI + 2));
    }
    return parts.map(([s, e]) => this.ranks.get(piece.slice(s, e)));
  }

  _encodeOrdinary(text) {
    const ids = [];
    PRETOKENIZER.lastIndex = 0;
    let m;
    while ((m = PRETOKENIZER.exec(text)) !== null) {
      if (m[0] === '') { PRETOKENIZER.lastIndex++; continue; }
      ids.push(...this._bytePairEncode(utf8ToBin(m[0])));
    }
    return ids;
  }

  encode(text, allowedSpecial = 'all') {
    const allowed = allowedSpecial === 'all'
      ? new Set(Object.keys(this.specialTokens)) : new Set();
    if (!this.specialRe || allowed.size === 0) return this._encodeOrdinary(text);
    const ids = [];
    let pos = 0, m;
    this.specialRe.lastIndex = 0;
    while ((m = this.specialRe.exec(text)) !== null) {
      if (m.index > pos) ids.push(...this._encodeOrdinary(text.slice(pos, m.index)));
      ids.push(this.specialTokens[m[0]]);
      pos = m.index + m[0].length;
    }
    if (pos < text.length) ids.push(...this._encodeOrdinary(text.slice(pos)));
    return ids;
  }

  decode(ids) {
    let bin = '';
    for (const id of ids) {
      const tok = this.decoder.get(id);
      if (tok === undefined) throw new Error(`Invalid token for decoding: ${id}`);
      bin += tok;
    }
    return decoder.decode(binToBytes(bin));
  }

  /** ids -> [{id, text}] for display (text may be undecodable alone). */
  tokensWithText(ids) {
    return ids.map(id => {
      const bin = this.decoder.get(id);
      let text;
      try { text = decoder.decode(binToBytes(bin)); }
      catch { text = '�'; }
      // lone surrogates / partial sequences show as replacement chars; keep raw
      return { id, text };
    });
  }
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/* ------------------------------------------------------------------ */
/* From-scratch BPE *training* (for the playground's Train tab).       */
/* Classic Sennrich-style: most frequent pair merges, applied in order.*/
/* ------------------------------------------------------------------ */

/** Train BPE on a corpus. Returns {merges: [[a,b]...], vocab: [strings]}. */
export async function trainBPE(corpus, numMerges, onProgress) {
  // word frequencies; spaces kept as their own pseudo-words
  const words = new Map();
  const pieces = corpus.match(/\S+|\s+/g) || [];
  for (const w of pieces) words.set(w, (words.get(w) || 0) + 1);

  // splits: word -> array of symbols
  const splits = new Map();
  for (const w of words.keys()) splits.set(w, [...w]);

  const merges = [];
  for (let m = 0; m < numMerges; m++) {
    const pairCounts = new Map();
    for (const [w, freq] of words) {
      const s = splits.get(w);
      for (let i = 0; i < s.length - 1; i++) {
        const p = s[i] + '\u0000' + s[i + 1];
        pairCounts.set(p, (pairCounts.get(p) || 0) + freq);
      }
    }
    if (pairCounts.size === 0) break;
    let best = null, bestC = -1;
    for (const [p, c] of pairCounts) {
      if (c > bestC) { bestC = c; best = p; }
    }
    const [a, b] = best.split('\u0000');
    merges.push([a, b]);
    const merged = a + b;
    for (const [w] of words) {
      const s = splits.get(w);
      const out = [];
      for (let i = 0; i < s.length; i++) {
        if (i < s.length - 1 && s[i] === a && s[i + 1] === b) {
          out.push(merged); i++;
        } else out.push(s[i]);
      }
      splits.set(w, out);
    }
    if (onProgress && m % 25 === 0) { onProgress(m / numMerges); await tick(); }
  }
  if (onProgress) { onProgress(1); await tick(); }
  const vocab = new Set();
  for (const s of splits.values()) for (const sym of s) vocab.add(sym);
  return { merges, vocab: [...vocab] };
}

/** Encode text with a trained merge list (greedy, in merge order). */
export function encodeWithMerges(text, merges) {
  const rank = new Map(merges.map(([a, b], i) => [a + '\u0000' + b, i]));
  const out = [];
  const pieces = text.match(/\S+|\s+/g) || [];
  for (const w of pieces) {
    let parts = [...w];
    for (;;) {
      let bestI = -1, bestR = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = rank.get(parts[i] + '\u0000' + parts[i + 1]);
        if (r !== undefined && r < bestR) { bestR = r; bestI = i; }
      }
      if (bestI === -1) break;
      parts = parts.slice(0, bestI).concat([parts[bestI] + parts[bestI + 1]], parts.slice(bestI + 2));
    }
    out.push(...parts);
  }
  return out;
}
