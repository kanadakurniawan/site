const sharp = require('sharp');
const path = require('path');

const src = 'C:/Users/Hi/Documents/Kanada/Riset/kanadakurniawan.com/kanada-kurniawan.webp';
const outDir = 'C:/Users/Hi/Documents/Kanada/Riset/kanadakurniawan.com/site/public/images';
const outFile = path.join(outDir, 'kanada-kurniawan.webp');

(async () => {
	require('fs').mkdirSync(outDir, { recursive: true });
	const meta = await sharp(src).metadata();
	const size = Math.min(meta.width, meta.height);
	await sharp(src)
		.resize(size, size, { fit: 'cover', position: 'top' })
		.webp({ quality: 82 })
		.toFile(outFile);
	const out = await sharp(outFile).metadata();
	const bytes = require('fs').statSync(outFile).size;
	console.log(
		JSON.stringify({
			in: { width: meta.width, height: meta.height },
			out: { width: out.width, height: out.height },
			bytes,
		}),
	);
})().catch((e) => {
	console.error(e);
	process.exit(1);
});