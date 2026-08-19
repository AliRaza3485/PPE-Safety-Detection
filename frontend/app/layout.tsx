import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PPE Safety Detection",
  description: "Detect hard hats, masks and safety vests in worksite images.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
