import type { APIRoute } from "astro";
import { renderOgJpeg } from "@/utils/og-template";
import config from "@/config";

export const GET: APIRoute = async context => {
  if (!config.features.dynamicOgImage) {
    return new Response(null, { status: 404, statusText: "Not found" });
  }

  const { buffer, contentType } = await renderOgJpeg(
    {
      title: config.site.title,
      subtitle: config.site.description,
      footerLeft: "Praktisi Meteorologi & Pembelajar Deep Learning",
      footerRight: new URL(config.site.url).hostname,
      // Keep the site-wide default stable and on-brand.
      variant: "split",
      palette: 0,
    },
    context.url
  );

  return new Response(new Uint8Array(buffer), {
    headers: {
      "Content-Type": contentType,
      // Browsers can cache aggressively; Satori/Sharp runs at build time so the
      // body is immutable until the next deploy.
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
};
