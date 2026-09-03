import sharp from "sharp";
import { mkdirSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, "..", "brand", "photo", "kanada-kurniawan.webp");
const out = join(root, "public", "favicon.ico");

const GRAD_A = "#08a88a";
const GRAD_B = "#1178c4";
const BORDER_RATIO = 0.12; // tebal border ~12% dari ukuran

async function buildIco(pngBuffers) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(pngBuffers.length, 4);
  const entries = [];
  let offset = 6 + 16 * pngBuffers.length;
  for (const { size, data } of pngBuffers) {
    const entry = Buffer.alloc(16);
    entry.writeUInt8(size >= 256 ? 0 : size, 0);
    entry.writeUInt8(size >= 256 ? 0 : size, 1);
    entry.writeUInt8(0, 2);
    entry.writeUInt8(0, 3);
    entry.writeUInt16LE(1, 4);
    entry.writeUInt16LE(32, 6);
    entry.writeUInt32LE(data.length, 8);
    entry.writeUInt32LE(offset, 12);
    entries.push(entry);
    offset += data.length;
  }
  return Buffer.concat([header, ...entries, ...pngBuffers.map((b) => b.data)]);
}

async function makeRoundedWithGradientRing(size) {
  const border = Math.max(1, Math.round(size * BORDER_RATIO));
  const inner = size - border * 2;

  // 1) Foto dipotong bulat (tanpa border)
  const photo = await sharp(src)
    .resize(inner, inner, { fit: "cover", position: "top" })
    .composite([
      {
        input: Buffer.from(
          `<svg width="${inner}" height="${inner}"><circle cx="${inner / 2}" cy="${inner / 2}" r="${inner / 2}" fill="white"/></svg>`
        ),
        blend: "dest-in",
      },
    ])
    .png()
    .toBuffer();

  // 2) Ring gradien (cincin) sebagai latar
  const ring = Buffer.from(
    `<svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="${GRAD_A}"/>
          <stop offset="1" stop-color="${GRAD_B}"/>
        </linearGradient>
      </defs>
      <circle cx="${size / 2}" cy="${size / 2}" r="${size / 2}" fill="url(#g)"/>
    </svg>`
  );

  // 3) Gabung: ring di bawah, foto bulat di tengah
  const final = await sharp(ring)
    .composite([
      {
        input: photo,
        left: border,
        top: border,
      },
    ])
    .png()
    .toBuffer();
  return final;
}

mkdirSync(dirname(out), { recursive: true });
const pngs = [];
for (const size of [16, 32]) {
  const data = await makeRoundedWithGradientRing(size);
  pngs.push({ size, data });
}
const ico = await buildIco(pngs);
await writeFile(out, ico);
console.log(`favicon.ico dibuat (bulat + ring gradien): ${out} (${ico.length} bytes)`);