// Where capture.mjs keeps a clip it cut, so film.mjs can find the same one.
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MEDIA = join(resolve(dirname(fileURLToPath(import.meta.url)), "..", ".."), "data", "cache", "media", "demo");
export const clipPath = (audio, start, seconds) =>
  join(MEDIA, `clip-${createHash("sha1").update(String(audio)).digest("hex").slice(0, 8)}-${start}-${seconds}.wav`);
