import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://kanadakurniawan.com",
    title: "Kanada Kurniawan",
    description: "Research on artificial intelligence and meteorology.",
    author: "Kanada Kurniawan",
    profile: "https://kanadakurniawan.com",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Pontianak",
    dir: "ltr",
  },
  posts: {
    perPage: 4,
    perIndex: 4,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: false,
    },
    search: "pagefind",
  },
  socials: [
    { name: "linkedin", url: "https://www.linkedin.com/in/kanadakurniawan/" },
    { name: "scholar", url: "https://scholar.google.com/citations?user=lLQWmDAAAAAJ&hl=en" },
    { name: "mail", url: "mailto:kanada.kurniawan@bmkg.go.id" },
  ],
  shareLinks: [
    { name: "whatsapp", url: "https://wa.me/?text=" },
    { name: "facebook", url: "https://www.facebook.com/sharer.php?u=" },
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "pinterest", url: "https://pinterest.com/pin/create/button/?url=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
