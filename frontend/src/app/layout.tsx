import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Animal Face CBIR",
  description:
    "Tìm kiếm ảnh dựa trên nội dung (CBIR) cho ảnh mặt động vật sử dụng đặc trưng thủ công.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi">
      <body className="bg-slate-50 text-slate-900 antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}
