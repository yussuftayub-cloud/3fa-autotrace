import Link from "next/link";

export default function Home() {
  return (
    <main>
      <nav className="nav">
        <div className="brand"><span>3FA</span> AUTO TRACE</div>
        <div className="navlinks">
          <a href="#features">Features</a>
          <a href="#pricing">Pricing</a>
          <Link href="/studio">AI Studio</Link>
          <Link className="navbutton" href="/studio">Try Free</Link>
        </div>
      </nav>
      <section className="hero">
        <div className="badge">✦ AI VECTOR PLATFORM FOR PRINTING</div>
        <h1>Turn your artwork into<br/><em>professional vectors.</em></h1>
        <p>Upload PNG, JPG or WebP. 3FA AUTO TRACE cleans, traces and prepares artwork for EPS, SVG and print workflows.</p>
        <div className="actions">
          <Link className="primary" href="/studio">Start Free →</Link>
          <a className="secondary" href="#pricing">View Pro RM49.90</a>
        </div>
        <div className="hero-card">
          <div className="mini-top"><span>AI Studio</span><span>● Ready</span></div>
          <div className="drop-demo"><div className="upload-icon">↑</div><strong>Drop artwork here</strong><small>PNG · JPG · WebP · Max 15MB</small></div>
        </div>
      </section>
      <section id="features" className="section">
        <div className="eyebrow">WHY 3FA</div>
        <h2>Built for designers & printing businesses.</h2>
        <div className="grid">
          {[
            ["✦", "AI Auto Trace", "Convert raster artwork into clean vector paths in seconds."],
            ["◇", "EPS / SVG Export", "Print-ready vector formats for your production workflow."],
            ["⚡", "Fast Processing", "Optimised processing flow for everyday design jobs."],
            ["◈", "AI Print Ready", "Quality checks and recommendations before export."]
          ].map(([icon,title,text]) => (
            <article className="feature" key={title}><div className="feature-icon">{icon}</div><h3>{title}</h3><p>{text}</p></article>
          ))}
        </div>
      </section>
      <section id="pricing" className="pricing">
        <div className="eyebrow">SIMPLE PRICING</div>
        <h2>One plan. Built for serious users.</h2>
        <div className="price-card">
          <div><span className="plan">PRO</span><h3>RM49.90 <small>/ month</small></h3><p>For designers, printers and businesses.</p></div>
          <ul><li>✓ AI Auto Trace</li><li>✓ EPS, SVG & PDF workflow</li><li>✓ AI image cleanup</li><li>✓ Cloud project history</li><li>✓ Priority processing</li></ul>
          <Link className="primary" href="/subscription">Start Pro →</Link>
        </div>
      </section>
      <footer>© 2026 3FA AUTO TRACE · Built for the printing industry.</footer>
    </main>
  );
}
