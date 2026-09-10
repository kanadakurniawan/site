import type { UIStrings } from "../types";

export default {
  nav: {
    home: "Beranda",
    posts: "Tulisan",
    tags: "Label",
    about: "Tentang",
    archives: "Arsip",
    search: "Cari",
    apps: "Aplikasi",
  },
  post: {
    publishedAt: "Dipublikasikan pada",
    updatedAt: "Diperbarui",
    sharePostIntro: "Bagikan tulisan ini:",
    sharePostOn: "Bagikan tulisan ini di {{platform}}",
    sharePostViaEmail: "Bagikan tulisan ini lewat email",
    tagLabel: "Label",
    backToTop: "Kembali ke atas",
    goBack: "Kembali",
    editPage: "Sunting halaman",
    previousPost: "Tulisan Sebelumnya",
    nextPost: "Tulisan Berikutnya",
  },
  pagination: {
    prev: "Sebelumnya",
    next: "Berikutnya",
    page: "Halaman",
  },
  home: {
    socialLinks: "Tautan Sosial",
    featured: "Unggulan",
    recentPosts: "Tulisan Terbaru",
    allPosts: "Semua Tulisan",
  },
  footer: {
    copyright: "Hak Cipta",
    allRightsReserved: "Hak cipta dilindungi undang-undang.",
  },
  pages: {
    tagTitle: "Label",
    tagDesc: "Semua artikel dengan label",

    tagsTitle: "Label",
    tagsDesc: "Daftar semua label yang dipakai dalam tulisan.",

    postsTitle: "Tulisan",
    postsDesc: "Semua artikel yang sudah saya terbitkan.",

    archivesTitle: "Arsip",
    archivesDesc: "Semua artikel yang sudah saya arsipkan.",

    searchTitle: "Cari",
    searchDesc: "Cari dalam artikel ...",

    appsTitle: "Aplikasi",
    appsDesc: "Kumpulan aplikasi yang saya buat.",
  },
  a11y: {
    skipToContent: "Lewati ke konten",
    openMenu: "Buka menu",
    closeMenu: "Tutup menu",
    toggleTheme: "Ubah tema",
    searchPlaceholder: "Cari artikel...",
    noResults: "Hasil tidak ditemukan",
    goToPreviousPage: "Ke halaman sebelumnya",
    goToNextPage: "Ke halaman berikutnya",
  },
  notFound: {
    title: "404 Tidak Ditemukan",
    message: "Halaman Tidak Ditemukan",
    goHome: "Kembali ke beranda",
  },
} satisfies UIStrings;