import { cpSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, "dist", "pagefind");
const dest = join(root, "public", "pagefind");

if (existsSync(src)) {
  cpSync(src, dest, { recursive: true });
  console.log("pagefind index disalin ke public/pagefind");
} else {
  console.warn("dist/pagefind tidak ditemukan — lewati penyalinan.");
}
