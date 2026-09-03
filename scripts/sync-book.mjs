import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const siteRoot = dirname(here);

const candidates = [
	join(siteRoot, '..', 'book', 'build', 'sync-to-blog.mjs'), // lokal: book/ sibling dari site/
	join(siteRoot, 'book', 'build', 'sync-to-blog.mjs'),       // CI: book/ di-checkout ke dalam repo site
];

const script = candidates.find(existsSync);
if (!script) {
	console.error('Tidak menemukan build/sync-to-blog.mjs di repo book.');
	console.error('Pastikan folder book tersedia sebagai sibling (lokal) atau sebagai ./book (CI).');
	process.exit(1);
}

const { syncAll } = await import(pathToFileURL(script).href);
console.log('Menjalankan sync buku -> blog...');
const n = syncAll();
console.log(`Selesai. ${n} bab disinkronkan.`);
