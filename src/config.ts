export const base = import.meta.env.BASE_URL;
export const siteUrl = `${import.meta.env.SITE}`.replace(/\/+$/, '');

export const siteConfig = {
	title: 'Kanada Kurniawan',
	description: 'Research on artificial intelligence and meteorology.',
	url: siteUrl,
	author: 'Kanada Kurniawan',
	locale: 'en',
	defaultOgg: `${base}og-default.png`,
	nav: [
		{ title: 'Home', href: `${base}` },
		{ title: 'Blog', href: `${base}blog/` },
		{ title: 'Publications', href: `${base}publications/` },
		{ title: 'About', href: `${base}about/` },
	],
} as const;