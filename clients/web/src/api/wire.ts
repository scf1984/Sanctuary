// The client's half of api.wire: a tagged JSON value in, a checked value out, and a Column's
// base64 bytes as a typed array. Every other shape is generated (./values.ts), so this file holds
// the only wire logic written by hand on this side.

import type { Column, Value } from "./values";

/** A Column's elements, ready to index. 64-bit integers arrive as doubles (see `decodeColumn`). */
export type Elements = Float32Array | Float64Array | Int8Array | Int16Array | Int32Array | Uint8Array | Uint16Array | Uint32Array;

export interface Decoded<E extends Elements = Elements> {
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly values: E;
}

// Typed arrays read in the platform's byte order and the wire is always little-endian. Every
// browser this game targets is little-endian, but a wrong order would decode every number into a
// plausible-looking wrong one, so it is checked once rather than assumed (CLAUDE.md §8.7).
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

const VIEWS: { readonly [dtype: string]: (buffer: ArrayBuffer) => Elements } = {
  "<f4": (b) => new Float32Array(b),
  "<f8": (b) => new Float64Array(b),
  "|i1": (b) => new Int8Array(b),
  "<i2": (b) => new Int16Array(b),
  "<i4": (b) => new Int32Array(b),
  "|u1": (b) => new Uint8Array(b),
  "|b1": (b) => new Uint8Array(b),
  "<u2": (b) => new Uint16Array(b),
  "<u4": (b) => new Uint32Array(b),
  // Ids are `<i8` (api.values.Frame). A double holds every integer below 2^53 exactly, which ids
  // will not reach, and indexing with a bigint would leak into every caller. Beyond 2^53 it refuses.
  "<i8": (b) => {
    const wide = new BigInt64Array(b);
    const out = new Float64Array(wide.length);
    for (let i = 0; i < wide.length; i++) {
      const value = wide[i] as bigint;
      if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
        throw new RangeError(`an <i8 element (${value}) does not fit a double exactly`);
      }
      out[i] = Number(value);
    }
    return out;
  },
};

/** A Column's bytes as a typed array, with its dtype and shape checked against each other. */
export function decodeColumn(column: Column): Decoded {
  if (!LITTLE_ENDIAN) {
    throw new Error("this client reads little-endian columns and the platform is big-endian");
  }
  const view = VIEWS[column.dtype];
  if (view === undefined) {
    throw new TypeError(`no typed array for dtype ${column.dtype}`);
  }
  const text = atob(column.data);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  const values = view(bytes.buffer);
  const expected = column.shape.reduce((n, d) => n * d, 1);
  if (values.length !== expected) {
    throw new RangeError(`a ${column.dtype} column of shape [${column.shape}] holds ${values.length} elements`);
  }
  return { dtype: column.dtype, shape: column.shape, values };
}

/** `value` if it carries the tag `name`, or a loud error: a route answering the wrong type is a
 * protocol bug, and drawing whatever arrived would hide it. */
export function expectType<N extends Value["$type"]>(value: unknown, name: N): Extract<Value, { $type: N }> {
  const tag = (value as { $type?: unknown } | null)?.$type;
  if (tag !== name) {
    throw new TypeError(`expected a ${name}, got ${JSON.stringify(tag ?? value)}`);
  }
  return value as Extract<Value, { $type: N }>;
}
