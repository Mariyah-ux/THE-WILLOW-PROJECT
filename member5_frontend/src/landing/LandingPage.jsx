import { HeroScene } from "./HeroScene.jsx";
import "./landing.css";

export function LandingPage({ onEnter }) {
  return (
    <main className="landing">
      <HeroScene />
      <section className="intro">
        <div className="intro__inner">
          <svg
            className="intro__ornament"
            viewBox="0 0 120 24"
            fill="none"
            aria-hidden="true"
          >
            <path d="M4 13 H 47" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
            <path d="M116 13 H 73" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
            <path
              d="M60 19 C 55.5 13.5, 55.5 6.5, 60 2 C 64.5 6.5, 64.5 13.5, 60 19 Z"
              fill="currentColor"
            />
            <path d="M60 19 V 22" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
          </svg>
          <h1 className="intro__title">The Willow Project</h1>
          <p className="intro__subtitle">
            A little space to pause, breathe, and put your thoughts into words.
          </p>
          <button type="button" className="intro__cta" onClick={onEnter}>
            Enter Willow
          </button>
        </div>
      </section>
    </main>
  );
}
