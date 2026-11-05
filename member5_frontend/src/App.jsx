import { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import {
  Home, ChartNoAxesColumnIncreasing, Clock3, Info, FileText, ShieldCheck,
  Sparkles, ArrowRight, Upload, Download, Heart, Menu, X, Check, Sun, Moon,
  Leaf, History, AlertCircle
} from "lucide-react";
import { predictText } from "./api";
import { normalizePrediction } from "./prediction-label.js";
import { LandingPage } from "./landing/LandingPage.jsx";

const THOUGHTS = [
  "You matter, even on the days when it is difficult to feel it.",
  "You do not have to face everything alone. Reaching out is a brave first step.",
  "One small step forward is still progress.",
  "Your feelings deserve to be heard, and support is available.",
  "You are worthy of care, patience, and kindness.",
  "A difficult moment is not the whole story of your life."
];

function Flower({ className = "" }) {
  return <span aria-hidden="true" className={`flower ${className}`}>
    <i /><i /><i /><i /><i /><b />
  </span>;
}

export default function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [text, setText] = useState("");
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [csvRows, setCsvRows] = useState([]);
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [textColumn, setTextColumn] = useState("");
  const [csvBusy, setCsvBusy] = useState(false);
  const [csvMessage, setCsvMessage] = useState("");
  const [active, setActive] = useState("Home");
  const [dark, setDark] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const fileRef = useRef(null);

  const thought = useMemo(() => THOUGHTS[new Date().getDate() % THOUGHTS.length], []);
  const navItems = [
    { name: "Home", icon: Home },
    { name: "Predictions", icon: ChartNoAxesColumnIncreasing },
    { name: "Latest result", icon: Clock3 },
    { name: "Responsible use", icon: Info }
  ];

  async function handlePredict(event) {
    event?.preventDefault();
    if (!text.trim()) {
      setError("Please enter some text before selecting Predict.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const result = await predictText(text.trim());
      const item = {
        ...result,
        display: normalizePrediction(result.prediction),
        modelDisplay: normalizePrediction(result.modelPrediction),
        sourceText: text.trim(),
        time: new Date()
      };
      setLatest(item);
      setHistory((old) => [item, ...old].slice(0, 20));
      setActive("Latest result");
    } catch (err) {
      setError(`${err.message}. Check that your Flask backend is running and /api/predict is available.`);
    } finally {
      setBusy(false);
    }
  }

  function handleCsvFile(file) {
    setCsvMessage("");
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setCsvMessage("Please choose a .csv file.");
      return;
    }
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: ({ data, meta, errors }) => {
        if (errors?.length) console.warn("CSV parsing warnings:", errors);
        const headers = meta.fields || [];
        if (!headers.length || !data.length) {
          setCsvMessage("The CSV appears to be empty or has no header row.");
          return;
        }
        setCsvHeaders(headers);
        setCsvRows(data);
        const preferred = headers.find(h => /text|thought|post|content|message/i.test(h)) || headers[0];
        setTextColumn(preferred);
        setCsvMessage(`${data.length} rows loaded. Choose the column containing the text, then run bulk prediction.`);
      }
    });
  }

  async function runCsvPrediction() {
    if (!csvRows.length || !textColumn) return;
    setCsvBusy(true);
    setCsvMessage("Predicting rows. Please keep this page open...");
    const output = [];
    let failed = 0;
    for (let i = 0; i < csvRows.length; i++) {
      const sourceText = String(csvRows[i][textColumn] ?? "").trim();
      if (!sourceText) {
        output.push({ ...csvRows[i], Willow_Prediction: "Skipped - empty text", Willow_Confidence: "" });
        continue;
      }
      try {
        const result = await predictText(sourceText);
        output.push({
          ...csvRows[i],
          Willow_Prediction: normalizePrediction(result.prediction),
          Willow_Model_Prediction: normalizePrediction(result.modelPrediction),
          Willow_Prediction_Source: result.predictionSource,
          Willow_Confidence: result.confidence === null ? "" : `${(result.confidence * 100).toFixed(1)}%`
        });
      } catch (err) {
        failed++;
        output.push({ ...csvRows[i], Willow_Prediction: "Error", Willow_Confidence: "", Willow_Error: err.message });
      }
    }
    setCsvRows(output);
    setCsvMessage(`Finished: ${output.length} rows processed${failed ? `; ${failed} errors` : ""}. Download the results below.`);
    setCsvBusy(false);
  }

  function downloadCsv() {
    const csv = Papa.unparse(csvRows);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "willow_prediction_results.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  function goTo(name) {
    setActive(name);
    setMobileMenu(false);
    const target = document.getElementById(name.toLowerCase().replaceAll(" ", "-"));
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  if (showLanding) {
    return <LandingPage onEnter={() => setShowLanding(false)} />;
  }

  return (
    <div className={dark ? "app dark" : "app"}>
      <button className="mobile-menu" onClick={() => setMobileMenu(!mobileMenu)} aria-label="Toggle navigation">
        {mobileMenu ? <X /> : <Menu />}
      </button>
      <aside className={`sidebar ${mobileMenu ? "show" : ""}`}>
        <div className="brand-mark"><Leaf size={34} /><span>WILLOW</span></div>
        <nav>
          {navItems.map(({ name, icon: Icon }) => (
            <button key={name} className={`nav-link ${active === name ? "selected" : ""}`} onClick={() => goTo(name)}>
              <Icon size={23} strokeWidth={2} /><span>{name}</span>
            </button>
          ))}
        </nav>
        <div className="side-divider"><span>✿</span></div>
        <p className="sidebar-quote">Better mental health<br />starts with<br />awareness.</p>
        <div className="sidebar-bottom">A compassionate AI-assisted tool</div>
        <span className="side-leaves">❧</span>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="top-brand">Willow <span>|</span> <small>Suicidal Ideation Thoughts Search</small></div>
          <div className="top-actions">
            <button title="Back to landing page" onClick={() => setShowLanding(true)}><Home size={20} /></button>
            <button title="Light mode" onClick={() => setDark(false)}><Sun size={20} /></button>
            <button title="Dark mode" onClick={() => setDark(true)}><Moon size={20} /></button>
            <button title="Responsible use" onClick={() => goTo("Responsible use")}><ShieldCheck size={21} /></button>
            <span className="top-sprig">✣</span>
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <div className="hero-title-row"><Leaf className="hero-leaf" size={46} /><div>
              <h1>WILLOW</h1>
              <h2>Suicidal Ideation Thoughts Search</h2>
            </div></div>
            <p>A kind and compassionate tool to identify possible signs of suicidal ideation in text,<br className="desktop-break" /> helping encourage awareness and support.</p>
            <div className="hero-tags"><span>• Analyze</span><span>• Understand</span><span>• Support</span></div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="hero-orb" /><Flower /><Flower className="small-flower" />
            <p>You are<br />not alone ♡</p>
          </div>
          <span className="corner-leaf">❧</span>
        </section>

        <div className="content-grid">
          <div className="left-column">
            <section className="panel text-panel" id="home">
              <div className="section-heading">
                <span className="heading-icon"><FileText size={26} /></span>
                <div><h3>Type Your Text</h3><p>Enter text below, then select Predict to check the model's classification.</p></div>
              </div>
              <form onSubmit={handlePredict}>
                <div className="textarea-wrap">
                  <textarea maxLength={1000} value={text} onChange={e => setText(e.target.value)} placeholder="e.g. Write or paste text here..." />
                  <div className="textarea-foot"><span>Keep in mind: a model prediction is not a diagnosis.</span><span>{text.length} / 1000</span></div>
                </div>
                <div className="predict-row">
                  <div className="thoughtful"><span className="check-icon"><Check size={19} /></span><div><b>Use thoughtfully</b><small>Only enter text you are permitted to analyze. Avoid personal identifiers.</small></div></div>
                  <button className="predict-btn" type="submit" disabled={busy}>{busy ? "Analyzing..." : <><Sparkles size={20} fill="currentColor" /> Predict <ArrowRight size={20} /></>}</button>
                </div>
              </form>
              {error && <p className="error-msg"><AlertCircle size={16} />{error}</p>}
            </section>

            <section className="panel csv-panel" id="predictions">
              <div className="section-heading">
                <span className="heading-icon peach"><Upload size={25} /></span>
                <div><h3>Upload CSV</h3><p>Bulk-check text rows from a CSV file and export the prediction results.</p></div>
              </div>
              <div className="upload-area" onClick={() => fileRef.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); handleCsvFile(e.dataTransfer.files?.[0]); }}>
                <Upload size={27} /><b>Choose a CSV file or drag it here</b><small>CSV with a header row; for example, a column named text, post, or content.</small>
                <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={e => handleCsvFile(e.target.files?.[0])} />
                <button type="button" className="browse-btn" onClick={e => { e.stopPropagation(); fileRef.current?.click(); }}>Browse files</button>
              </div>
              {csvHeaders.length > 0 && <div className="csv-controls">
                <label>Text column
                  <select value={textColumn} onChange={e => setTextColumn(e.target.value)}>
                    {csvHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </label>
                <span>{csvRows.length} rows</span>
                <button className="predict-btn csv-run" disabled={csvBusy} onClick={runCsvPrediction}>{csvBusy ? "Processing..." : "Predict all rows"} <ArrowRight size={17} /></button>
              </div>}
              {csvMessage && <p className="csv-message">{csvMessage}</p>}
              {csvRows.some(r => "Willow_Prediction" in r) && <button className="download-btn" onClick={downloadCsv}><Download size={17} /> Download prediction CSV</button>}
            </section>

            <section className="panel latest-panel" id="latest-result">
              <div className="section-heading latest-heading">
                <span className="heading-icon yellow"><ChartNoAxesColumnIncreasing size={25} /></span>
                <div><h3>Latest Prediction</h3><p>{latest ? "Your latest response from the connected model." : "Your latest response from the connected model appears here."}</p></div>
              </div>
              {latest ? <div className="result-grid">
                <div className="result-card green-result">
                  <small>{latest.predictionSource === "explicit_language" ? "EXPLICIT LANGUAGE FLAG" : "MODEL PREDICTION"}</small>
                  <strong>{latest.display}</strong>
                </div>
                <div className="result-card pink-result">
                  {latest.predictionSource === "explicit_language" ? (
                    <>
                      <small>MODEL PREDICTION</small>
                      <strong>{latest.modelDisplay}</strong>
                    </>
                  ) : (
                    <>
                      <small>MODEL CONFIDENCE</small>
                      <strong>{latest.confidence === null ? "Not provided" : `${(latest.confidence * 100).toFixed(1)}%`}</strong>
                    </>
                  )}
                </div>
              </div> : <div className="empty-result">Your prediction will appear here after you analyze text.</div>}
            </section>

            <section className="panel history-panel">
              <div className="section-heading"><span className="heading-icon"><History size={24} /></span><div><h3>Recent Predictions</h3><p>Recent results from this browser session.</p></div></div>
              {history.length ? <div className="history-list">{history.slice(0, 5).map((item, i) => <div className="history-item" key={`${item.time.getTime()}-${i}`}><span>{item.display}</span><small>{item.time.toLocaleTimeString()}</small></div>)}</div> : <p className="empty-result">No predictions yet.</p>}
            </section>
          </div>

          <aside className="right-column">
            <section className="panel responsible-panel" id="responsible-use">
              <div className="section-heading"><span className="heading-icon shield"><ShieldCheck size={25} fill="currentColor" /></span><div><h3>Responsible Use</h3></div></div>
              <ul className="responsible-list">
                <li><Check />This tool is for educational and supportive purposes only.</li>
                <li><Check />It cannot diagnose a person or replace professional mental health care.</li>
                <li><Check />Predictions can be wrong. Do not use them as the sole basis for decisions about someone's safety.</li>
              </ul>
              <div className="support-block"><span className="tiny-flower">✿</span><h4>Need support?</h4><p>If you or someone else may be in immediate danger, contact local emergency services or a trusted person now.</p>
                <a className="support-btn" href="https://112.gov.in/" target="_blank" rel="noreferrer">Find support resources <ArrowRight size={16} /></a>
                <p className="india-note">In India, call 112 for emergency assistance. For mental health support, Tele-MANAS is available at 14416.</p>
              </div>
              <Flower className="corner-flower" />
            </section>

            <section className="panel thought-panel">
              <div className="thought-top"><span className="heading-icon peach"><Heart size={23} /></span><span className="eyebrow">A GENTLE REMINDER</span></div>
              <h3>Today's Thought</h3>
              <p className="daily-thought">“{thought}”</p>
              <span className="thought-sprig">— with care, Willow</span>
            </section>
          </aside>
        </div>
        <footer className="footer">Willow · A compassionate AI-assisted awareness tool <span>Made with care ♡</span></footer>
      </main>
    </div>
  );
}