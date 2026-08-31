// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Di GitHub Pages project repo, base harus '/<repo-name>/'
// Saat nanti pakai custom domain, ubah jadi: site: 'https://kanadakurniawan.com' + base: '/'
export default defineConfig({
	site: 'https://kanadakurniawan.github.io',
	base: '/site/',
	markdown: {
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
	},
	integrations: [mdx(), sitemap()],
});