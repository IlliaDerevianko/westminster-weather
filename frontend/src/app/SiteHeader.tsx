import Link from "next/link";
export default function SiteHeader() {
    return <header className="site-header"><div className="site-header-inner">
    <Link href="/bills" aria-label="Westminster Weather home" className="brand">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/brand/logo-horizontal-full-colour.svg" alt="Westminster Weather"/>
    </Link><nav aria-label="Main navigation"><Link href="/bills">Government bills</Link></nav>
  </div></header>;
}
