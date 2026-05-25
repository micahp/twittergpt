import "./globals.css";
import Link from "next/link";

export const metadata = { title: "twitterGPT", description: "Finetune an LLM on your tweets" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav>
          <Link href="/"><b>twitterGPT</b></Link>
          <Link href="/">1. Upload</Link>
          <Link href="/review">2. Review</Link>
          <Link href="/train">3. Train</Link>
          <Link href="/generate">4. Generate</Link>
        </nav>
        <div className="wrap">{children}</div>
      </body>
    </html>
  );
}
