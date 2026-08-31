import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { getCollection } from 'astro:content';
import { base, siteConfig } from '../config';

export async function GET(context: APIContext) {
	const posts = (await getCollection('posts', ({ data }) => !data.draft)).sort(
		(a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf(),
	);

	return rss({
		title: siteConfig.title,
		description: siteConfig.description,
		site: `${siteConfig.url}${base}`.replace(/\/+$/, ''),
		items: posts.map((post) => ({
			title: post.data.title,
			description: post.data.description,
			pubDate: post.data.pubDate,
			link: `${base}blog/${post.id}/`,
		})),
		customData: `<language>${siteConfig.locale}</language>`,
	});
}