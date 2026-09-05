// Generate the brand profile image used by OG templates.
// Produces: src/assets/images/brand/profile.png
// - square center-crop of the source photo
// - circular clip
// - gradient ring border matching --accent-grad (teal -> blue)
// - 320x320 (drawn at ~280px in the 1200x630 OG card)
//
// The circular crop + ring are baked into a single SVG and rasterised by
// librsvg (via sharp). Emoji/clip handling is deterministic and avoids the
// confusion of multi-step raw compositing.
import sharp from "sharp";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const SRC = path.join(root, "public/images/kanada-kurniawan.webp");
const OUT = path.join(root, "src/assets/images/brand/profile.png");

const SIZE = 320;
const STROKE = 10;
const RADIUS = SIZE / 2 - STROKE / 2; // 155px

// Accent gradient colours from theme.css (light variant).
const COLOR_INNER = "#08a88a"; // accent-1 (teal)
const COLOR_OUTER = "#1178c4"; // accent-2 (blue)

async function main() {
  const raw = await fs.readFile(SRC);

  // 1. Square center-crop the source photo.
  const face = await sharp(raw)
    .resize(SIZE, SIZE, { fit: "cover", position: "centre" })
    .png()
    .toBuffer();

  // 2. One SVG does the circular clip + image + gradient ring.
  const svg = `<svg width="${SIZE}" height="${SIZE}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="circle">
      <circle cx="${SIZE / 2}" cy="${SIZE / 2}" r="${RADIUS}" />
    </clipPath>
    <linearGradient id="ring" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${COLOR_INNER}" />
      <stop offset="100%" stop-color="${COLOR_OUTER}" />
    </linearGradient>
  </defs>
  <image
    href="data:image/png;base64,${face.toString("base64")}"
    x="0" y="0" width="${SIZE}" height="${SIZE}"
    preserveAspectRatio="xMidYMid slice"
    clip-path="url(#circle)"
  />
  <circle
    cx="${SIZE / 2}" cy="${SIZE / 2}" r="${RADIUS}"
    fill="none" stroke="url(#ring)" stroke-width="${STROKE}"
  />
</svg>`;

  const final = await sharp(Buffer.from(svg))
    .png({ compressionLevel: 9, palette: true, quality: 80 })
    .toBuffer();

  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.writeFile(OUT, final);
  console.log(`Wrote ${OUT} (${(final.length / 1024).toFixed(1)} KB)`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});