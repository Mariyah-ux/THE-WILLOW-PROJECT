import { useEffect, useRef } from "react";
import willowScene from "../assets/willow-scene.jpg";
import willowLeaf from "../assets/willow-leaf.png";

const LEAF_COUNT = 6;

export function HeroScene() {
  const heroRef = useRef(null);

  useEffect(() => {
    const hero = heroRef.current;
    if (!hero) return;

    const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!finePointer.matches || reducedMotion.matches) return;

    let raf = 0;
    const onMove = (event) => {
      const rect = hero.getBoundingClientRect();
      const mx = Math.max(-1, Math.min(1, ((event.clientX - rect.left) / rect.width) * 2 - 1));
      const my = Math.max(-1, Math.min(1, ((event.clientY - rect.top) / rect.height) * 2 - 1));
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        hero.style.setProperty("--mx", mx.toFixed(3));
        hero.style.setProperty("--my", my.toFixed(3));
      });
    };
    const onLeave = () => {
      hero.style.setProperty("--mx", "0");
      hero.style.setProperty("--my", "0");
    };

    hero.addEventListener("pointermove", onMove);
    hero.addEventListener("pointerleave", onLeave);
    return () => {
      hero.removeEventListener("pointermove", onMove);
      hero.removeEventListener("pointerleave", onLeave);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <section
      ref={heroRef}
      className="hero"
      aria-label="An illustrated riverside willow grove in warm evening light"
    >
      <div className="hero__frame">
        <img
          className="hero__image"
          src={willowScene}
          width={1920}
          height={1088}
          fetchPriority="high"
          alt="A giant weeping willow beside a calm river at golden hour, with a person sitting peacefully beneath its hanging branches, wildflowers and mossy stones in the foreground"
        />
        <div className="hero__layer hero__layer--canopy" aria-hidden="true">
          <img src={willowScene} alt="" loading="lazy" className="hero__slice hero__slice--canopy" />
        </div>
        <div className="hero__layer hero__layer--sway-left" aria-hidden="true">
          <img src={willowScene} alt="" loading="lazy" className="hero__slice hero__slice--sway-left" />
        </div>
        <div className="hero__layer hero__layer--sway-center" aria-hidden="true">
          <img src={willowScene} alt="" loading="lazy" className="hero__slice hero__slice--sway-center" />
        </div>
        <div className="hero__layer hero__layer--sway-right" aria-hidden="true">
          <img src={willowScene} alt="" loading="lazy" className="hero__slice hero__slice--sway-right" />
        </div>
        <div className="hero__haze" aria-hidden="true" />
        <div className="hero__shimmer" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>
        <div className="hero__layer hero__layer--foreground" aria-hidden="true">
          <img src={willowScene} alt="" loading="lazy" className="hero__slice hero__slice--foreground" />
        </div>
        <div className="hero__leaves" aria-hidden="true">
          {Array.from({ length: LEAF_COUNT }, (_, index) => (
            <span
              key={index}
              className="hero__leaf"
              style={{ backgroundImage: `url(${willowLeaf})` }}
            />
          ))}
        </div>
        <div className="hero__light" aria-hidden="true" />
        <div className="hero__blend" aria-hidden="true" />
      </div>
    </section>
  );
}
