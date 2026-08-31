import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
	loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/posts' }),
	schema: z.object({
		title: z.string(),
		description: z.string(),
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		categories: z.array(z.string()).default([]),
		tags: z.array(z.string()).default([]),
		draft: z.boolean().default(false),
	}),
});

const publications = defineCollection({
	loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/publications' }),
	schema: z.object({
		title: z.string(),
		venue: z.string(),
		year: z.coerce.number(),
		doi: z.string().optional(),
		url: z.string().optional(),
		citation: z.string().optional(),
	}),
});

export const collections = { posts, publications };