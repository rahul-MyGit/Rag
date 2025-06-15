import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agency RAG Chat",
  description: "AI-powered chat for agency policies and client transcripts",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
