import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Clusture AI | Brain MRI Segmentation",
  description: "Experimental AI-assisted brain MRI segmentation showcase.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
