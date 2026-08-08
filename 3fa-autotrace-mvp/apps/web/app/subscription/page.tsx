import Link from "next/link";

export default function Subscription() {
  return <main>
    <nav className="nav"><div className="brand"><span>3FA</span> AUTO TRACE</div><Link className="navbutton" href="/studio">AI Studio</Link></nav>
    <section className="subscription">
      <div className="eyebrow">PRO PLAN</div>
      <h1>Everything you need to vectorise artwork.</h1>
      <div className="price-card">
        <span className="plan">PRO</span>
        <h2>RM49.90 <small>/ month</small></h2>
        <ul><li>✓ AI Auto Trace</li><li>✓ EPS & SVG export</li><li>✓ AI cleanup workflow</li><li>✓ Project history</li><li>✓ Priority processing</li></ul>
        <button className="primary wide" disabled>Payment Gateway — Connect Next</button>
        <p className="notice">Recurring billing is disabled until payment gateway credentials are connected. No real payment is collected by this MVP.</p>
      </div>
    </section>
  </main>;
}
