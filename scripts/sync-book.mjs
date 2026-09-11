import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const siteRoot = dirname(here);

// Satu buku = satu repo publik. Tambahkan entri di sini saat buku ke-2 ada.
const candidates = [
	join(siteRoot, '..', 'books', '01-pengantar-dl-meteorologi', 'build', 'sync-to-blog.mjs'), // lokal: books/<book>/ sibling dari site/
	join(siteRoot, 'book-01-pengantar-dl-meteorologi', 'build', 'sync-to-blog.mjs'),           // CI: repo buku di-checkout ke dalam repo site
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
