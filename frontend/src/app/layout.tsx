import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Westminster Weather — UK Government Bill Tracker",
  description:
    "Predicts whether UK Government Bills will pass and estimates the time until commencement.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
