import Link from "next/link";

// Site-wide header with the Westminster Weather logo
export default function SiteHeader() {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-20 flex items-center justify-between">
        <Link href="/" aria-label="Westminster Weather home" className="flex items-center">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/brand/logo-horizontal-full-colour.svg"
            alt="Westminster Weather"
            className="h-16 w-auto"
          />
        </Link>
        <nav className="flex items-center gap-2 text-sm font-semibold">
          <Link href="/" className="rounded-lg px-3 py-2 text-slate-700 hover:bg-slate-100">
            Tracker
          </Link>
          <Link href="/bills" className="rounded-lg px-3 py-2 text-slate-700 hover:bg-slate-100">
            All bills
          </Link>
        </nav>
      </div>
    </header>
  );
}
