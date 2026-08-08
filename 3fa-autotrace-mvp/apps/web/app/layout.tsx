import "./globals.css";

export const metadata = {
  title: "3FA AUTO TRACE",
  description: "AI-powered vector platform for printing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ms">
      <body>{children}</body>
    </html>
  );
}
