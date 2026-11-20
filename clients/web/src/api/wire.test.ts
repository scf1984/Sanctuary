import { describe, expect, it } from "vitest";

import type { Column } from "./values";
import { decodeColumn, expectType } from "./wire";

function column(dtype: string, shape: number[], bytes: Uint8Array): Column {
  let text = "";
  for (const byte of bytes) text += String.fromCharCode(byte);
  return { $type: "Column", dtype, shape, data: btoa(text) };
}

const le = (array: ArrayBufferView) => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);

describe("decodeColumn", () => {
  it("reads little-endian floats as the numbers that were packed", () => {
    const decoded = decodeColumn(column("<f4", [3], le(new Float32Array([1.5, -2, 1e6]))));
    expect(decoded.values).toBeInstanceOf(Float32Array);
    expect(Array.from(decoded.values)).toEqual([1.5, -2, 1e6]);
  });

  it("reads 64-bit ids as exact doubles", () => {
    const decoded = decodeColumn(column("<i8", [2], le(new BigInt64Array([7n, 9_007_199_254_740_991n]))));
    expect(Array.from(decoded.values)).toEqual([7, 9_007_199_254_740_991]);
  });

  it("refuses an id a double cannot hold exactly", () => {
    expect(() => decodeColumn(column("<i8", [1], le(new BigInt64Array([2n ** 60n]))))).toThrow(RangeError);
  });

  it("reads booleans as bytes", () => {
    expect(Array.from(decodeColumn(column("|b1", [2, 2], new Uint8Array([1, 0, 0, 1]))).values)).toEqual([1, 0, 0, 1]);
  });

  it("refuses a shape that does not match the bytes", () => {
    expect(() => decodeColumn(column("<f4", [4], le(new Float32Array([1, 2]))))).toThrow(RangeError);
  });

  it("refuses a dtype it has no array for", () => {
    expect(() => decodeColumn(column("<c16", [0], new Uint8Array()))).toThrow(TypeError);
  });
});

describe("expectType", () => {
  it("passes a value carrying the expected tag", () => {
    const status = { $type: "Status", tick: 3, collapsed: false, living: 10 };
    expect(expectType(status, "Status")).toBe(status);
  });

  it("refuses a value carrying another tag, rather than drawing it", () => {
    expect(() => expectType({ $type: "Frame" }, "Status")).toThrow(/expected a Status/);
    expect(() => expectType(null, "Status")).toThrow(TypeError);
  });
});
