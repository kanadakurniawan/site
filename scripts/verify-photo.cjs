const fs = require('fs');

for (const p of ['dist/index.html', 'dist/about/index.html']) {
	const c = fs.readFileSync(p, 'utf8');
	console.log(`${p}: avatar=${c.includes('/images/kanada-kurniawan.webp')}`);
}
const stat = fs.statSync('public/images/kanada-kurniawan.webp');
console.log('processed photo bytes:', stat.size);