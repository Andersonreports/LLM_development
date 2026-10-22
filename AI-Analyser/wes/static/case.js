/* Case page: status polling, review editor, candidate browser.
   Not every element exists on every page state (processing/failed pages are minimal,
   the finalize/evidence elements only exist once status is review/approved/rejected),
   so every DOM write goes through setHTML/setText, which no-op when the target is absent. */
(() => {
  const C = window.CASE;
  const $ = (id) => document.getElementById(id);
  const setHTML = (id, html) => { const el = $(id); if (el) el.innerHTML = html; };
  const setText = (id, text) => { const el = $(id); if (el) el.textContent = text; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const SHORT = {"Pathogenic": "P", "Likely pathogenic": "LP", "Uncertain significance": "VUS", "Likely benign": "LB", "Benign": "B"};
  const ROLE = {proband: "Proband", father: "Father", mother: "Mother", partner1: "P1", partner2: "P2"};
  const ZYG = {het: "Het", hom: "Hom", hemizygous: "Hemi", ref: "Ref", no_call: "NC", homoplasmic: "Homo", heteroplasmic: "Hetero"};
  let data = null, review = null, pool = {};

  // ---------------------------------------------------------------- page tabs (overview/patient/evidence/activity)
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll("[data-pane]").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== b.dataset.tab));
  }));
  if (C.type !== "couple") document.querySelectorAll(".couple-only").forEach((e) => e.classList.add("hidden"));

  // ---------------------------------------------------------------- reject flow
  if ($("rejectBtn")) {
    $("rejectBtn").addEventListener("click", () => $("rejectForm").classList.remove("hidden"));
    $("cancelReject").addEventListener("click", () => $("rejectForm").classList.add("hidden"));
  }

  // ---------------------------------------------------------------- pipeline tracker
  const ICON = {done: "&#10003;", failed: "&#10005;", skipped: "&ndash;", current: "", pending: ""};
  const fmtDur = (s) => s == null ? "" : s < 60 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  function renderSteps(steps) {
    setHTML("stepper", steps.map((st, i) => `
      <div class="ps ${st.state === "done" || st.state === "skipped" ? "done" : st.state === "current" ? "cur" : st.state === "failed" ? "fail" : ""}" title="${esc(st.desc)}">
        <div class="b">${st.state === "pending" ? i + 1 : ICON[st.state]}</div>${esc(st.label)}</div>`).join(""));
    const box = $("pipeline");
    if (!box) return;
    const rows = steps.filter((st) => !["review", "approved"].includes(st.key));
    box.innerHTML = rows.map((st) => {
      const live = st.state === "current" && st.started ? (Date.now() - new Date(st.started).getTime()) / 1000 : null;
      const right = st.state === "done" ? `<span class="small muted">${fmtDur(st.seconds)}</span>`
        : st.state === "current" ? `<span class="small" style="color:var(--brand)"><span class="spinner"></span> running ${live != null ? fmtDur(live) : ""}</span>`
        : st.state === "failed" ? '<span class="pill st st-failed">failed</span>'
        : st.state === "skipped" ? '<span class="small muted">skipped</span>' : '<span class="small faint">waiting</span>';
      const dot = {done: "background:var(--brand);color:#fff;border-color:var(--brand)", current: "border-color:var(--brand);color:var(--brand)",
        failed: "background:var(--pos);border-color:var(--pos);color:#fff", skipped: "background:#eef2f6;color:var(--muted)", pending: ""}[st.state];
      return `<div style="display:flex;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line-2)">
        <div style="width:26px;height:26px;border-radius:50%;border:2px solid var(--line);display:grid;place-items:center;font-size:12px;font-weight:700;flex:none;${dot}">${st.state === "current" ? '<span class="spinner" style="width:11px;height:11px"></span>' : st.state === "pending" ? "" : ICON[st.state]}</div>
        <div style="flex:1;min-width:0"><div style="font-weight:600;${st.state === "pending" ? "color:var(--faint)" : ""}">${esc(st.label)}</div><div class="xs muted">${esc(st.desc)}</div></div>
        <div>${right}</div></div>`;
    }).join("");
    const first = steps.find((st) => st.started);
    if (first && ["processing", "queued"].includes(C.status)) setText("pipeElapsed", `Elapsed ${fmtDur((Date.now() - new Date(first.started).getTime()) / 1000)}`);
    const current = steps.find((st) => st.state === "current");
    if (current) setText("currentStepLabel", current.label + "…");
    else if (C.status === "queued") setText("currentStepLabel", "Queued…");
  }
  renderSteps(C.steps);

  // older log entries may carry raw technical wording from before this UI existed;
  // clean those up for display without needing to touch the stored data
  function friendlyLog(msg) {
    return msg
      .replace(/\(no ANTHROPIC_API_KEY\)/g, "(no API key set)")
      .replace(/ANTHROPIC_API_KEY/g, "API key")
      .replace(/^AI interpretation unavailable.*$/, "AI assistant is not configured (no API key set) - a rule-based draft was prepared instead.")
      .replace(/^AI interpretation failed: (.+) - draft built from rules$/, "AI assistant could not complete interpretation ($1) - a rule-based draft was prepared instead.");
  }

  // ---------------------------------------------------------------- status polling
  async function poll() {
    const r = await fetch(`/api/cases/${C.id}/status`);
    if (!r.ok) return;
    const s = await r.json();
    renderSteps(s.steps);
    // "stage" events just mark progress through a step - that's already shown by the
    // pipeline tracker above, so the activity log only needs the human-readable events
    const visible = s.events.filter((e) => e.kind !== "stage").slice().reverse();
    setHTML("events", visible.length ? visible.map((e) => {
      const msg = friendlyLog(e.message.split("\n")[0]);
      return `<div class="log-row ${esc(e.kind)}"><div class="tag"></div><div class="body"><div class="m">${esc(msg).slice(0, 400)}</div>
        <div class="w">${new Date(e.at).toLocaleString([], {day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"})}${e.user ? " &middot; " + esc(e.user) : ""}</div></div></div>`;
    }).join("") : '<div class="log-empty">Nothing to show yet.</div>');
    if (s.status !== C.status) { location.reload(); return; }
    if (["processing", "queued"].includes(s.status)) setTimeout(poll, 2500);
  }
  poll();

  // ---------------------------------------------------------------- HPO editor
  const hpo = new Map((C.hpo || []).map((t) => [t.id, t]));
  function drawHpo() {
    const el = $("hpoChips");
    if (!el) return;
    el.innerHTML = [...hpo.values()].map((t) => `<span class="chip ${t.excluded ? "ex" : ""}">${esc(t.name)} <span class="muted">${esc(t.id)}</span>${C.hpoEditable ? `<button type="button" data-id="${esc(t.id)}">&times;</button>` : ""}</span>`).join("") || '<span class="muted small">None</span>';
  }
  async function saveHpo() {
    await fetch(`/api/cases/${C.id}/hpo`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({terms: [...hpo.values()]})});
    poll();
  }
  if ($("hpoChips")) {
    drawHpo();
    $("hpoChips").addEventListener("click", (e) => { if (e.target.dataset.id) { hpo.delete(e.target.dataset.id); drawHpo(); saveHpo(); } });
    let t;
    $("hpoSearch").addEventListener("input", () => {
      clearTimeout(t);
      const q = $("hpoSearch").value.trim();
      if (q.length < 3) { $("hpoList").classList.add("hidden"); return; }
      t = setTimeout(async () => {
        const hits = await (await fetch(`/api/hpo?q=${encodeURIComponent(q)}`)).json();
        setHTML("hpoList", hits.map((h) => `<div data-id="${esc(h.id)}" data-name="${esc(h.name)}">${esc(h.name)} <span class="muted small">${esc(h.id)}</span></div>`).join(""));
        $("hpoList").classList.remove("hidden");
      }, 200);
    });
    $("hpoList").addEventListener("click", (e) => {
      const d = e.target.closest("[data-id]");
      if (!d) return;
      hpo.set(d.dataset.id, {id: d.dataset.id, name: d.dataset.name, excluded: false});
      $("hpoSearch").value = ""; $("hpoList").classList.add("hidden"); drawHpo(); saveHpo();
    });
  }

  // ---------------------------------------------------------------- helpers
  const fmtAf = (af) => af == null ? "absent" : af === 0 ? "0" : af < 0.0001 ? af.toExponential(1) : (af * 100).toFixed(3) + "%";
  const crit = (c) => `<span class="crit ${c.code[0]} ${c.source === "ai" ? "ai" : ""}" title="${esc(c.reason)} [${esc(c.source)}]">${esc(c.code)}${c.strength && !defaultStrength(c) ? "_" + esc(c.strength) : ""}</span>`;
  function defaultStrength(c) {
    const d = {PVS: "VeryStrong", PS: "Strong", PM: "Moderate", PP: "Supporting", BA: "StandAlone", BS: "Strong", BP: "Supporting"};
    return d[c.code.startsWith("PVS") ? "PVS" : c.code.slice(0, 2)] === c.strength;
  }
  const zyg = (v) => Object.entries(v.zygosity || {}).map(([r, z]) => `${ROLE[r] || r}: ${ZYG[z] || z}`).join("<br>");
  const varLabel = (v) => `<span class="gene" style="font-style:italic;font-weight:700">${esc(v.gene)}</span> <span class="mono">${esc(v.hgvsc || v.variant_id)}</span> <span class="mono">${esc(v.hgvsp || "")}</span>`;

  // ---------------------------------------------------------------- review editor
  function renderReview() {
    if (!$("review")) return;
    if (!review) { setHTML("review", '<p class="muted">No results yet.</p>'); return; }
    const ro = !C.editable ? "disabled" : "";
    const flags = C.flags.filter((f) => C.type === "couple" || f !== "CARRIER");
    const attentionItems = review.reviewer_attention || [];
    const attention = attentionItems.length
      ? `<div class="attn-head">Needs your attention before approving (${attentionItems.length})</div>${attentionItems.map((a) => `<div class="warn">${esc(a)}</div>`).join("")}`
      : "";
    const srcLine = review.source === "ai"
      ? "This draft was written by the AI assistant from the evidence below."
      : "This draft was built automatically from the classification rules (no AI assistant was used for this case).";
    const disagrees = data.rule_flag && data.ai_result && data.rule_flag.flag !== data.ai_result.result_flag;
    const disagree = disagrees
      ? `<div class="warn">The automatic rule-based check suggests <b>${esc(data.rule_flag.flag)}</b>, while the AI assistant suggests <b>${esc(data.ai_result.result_flag)}</b>. Please review the evidence carefully before deciding.</div>` : "";
    if (C.editable) {
      const clean = attentionItems.length === 0 && !disagrees;
      setHTML("readyBanner", clean
        ? `<div class="ready-banner">${iconCheck()}No automatic exceptions were raised - review the findings below and confirm.</div>`
        : `<div class="ready-banner" style="background:var(--inc-bg);border-color:var(--inc-line);color:var(--inc)">${iconAlert()}${attentionItems.length + (disagrees ? 1 : 0)} item(s) need your attention before approving - see below.</div>`);
    }
    setHTML("review", `
      <div class="small muted" style="margin-bottom:8px">${srcLine}</div>
      ${disagree}
      ${attention}
      <div class="grid g3" style="margin-top:8px">
        <div><label>Result flag</label><select id="rvFlag" ${ro}>${flags.map((f) => `<option ${f === review.flag ? "selected" : ""}>${f}</option>`).join("")}</select></div>
        <div style="grid-column: span 2"><label>Summary line</label><input type="text" id="rvSummary" value="${esc(review.summary_line)}" ${ro}></div>
      </div>
      <h3>Report text</h3>
      <label>Clinical history (printed in the report)</label>
      <textarea id="rvHistory" rows="3" ${ro}>${esc(review.clinical_history_text)}</textarea>
      ${C.type !== "couple" ? `<div style="margin-top:10px"><label>CNV findings</label><textarea id="rvCnv" rows="2" style="min-height:56px" ${ro}>${esc(review.cnv_text)}</textarea></div>` : ""}
      <h3>Findings for report</h3>
      <div id="rvFindings"></div>
      <h3>Interpretation</h3>
      <textarea id="rvInterp" rows="6" ${ro}>${esc(review.clinical_interpretation)}</textarea>
      <div class="hint">Printed for positive, inconclusive and carrier results. Negative reports use the standard result statement.</div>
      <div class="grid g2" style="margin-top:10px">
        <div><label>Recommendations (one per line)</label><textarea id="rvRecs" rows="6" ${ro}>${esc((review.recommendations || []).join("\n"))}</textarea></div>
        <div><label>Case-specific limitations (one per line)</label><textarea id="rvLims" rows="6" ${ro}>${esc((review.limitations || []).join("\n"))}</textarea></div>
      </div>
      ${C.type !== "couple" ? `<h3>Genes analysed (Appendix 1)</h3><textarea id="rvGenes" rows="3" ${ro}>${esc((review.gene_list || (data.gene_list || [])).join(", "))}</textarea><div class="hint">Comma-separated. Defaults to genes matching the phenotype${data.has_coverage ? "; coverage % comes from the uploaded file" : "; no coverage file was uploaded, so coverage is not printed"}.</div>` : ""}
      ${C.editable ? `<div class="save-row"><button id="rvSave">Save changes</button><span class="small muted" id="rvMsg"></span></div>` : ""}`);
    renderFindings();
    if (C.editable && $("rvSave")) $("rvSave").addEventListener("click", saveReview);
  }
  function iconCheck() { return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>'; }
  function iconAlert() { return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/></svg>'; }

  function renderFindings() {
    const el = $("rvFindings");
    if (!el) return;
    if (!review.findings.length) { el.innerHTML = '<p class="muted small">No variants selected for the report. Add from "Supporting evidence" below if needed.</p>'; return; }
    const ro = !C.editable ? "disabled" : "";
    el.innerHTML = review.findings.map((f, i) => {
      const v = pool[f.variant_key] || {};
      const cls = f.classification_override || f.acmg.classification;
      return `<div class="finding sec-${esc(f.report_section)} ${f.include === false ? "off" : ""}" data-i="${i}">
        <div class="head">
          <label class="check"><input type="checkbox" data-k="include" ${f.include !== false ? "checked" : ""} ${ro}> include</label>
          ${varLabel(v)}
          <span class="small">${zyg(v)}</span>
          <span class="cls-${SHORT[cls]}">${esc(cls)}</span> <span class="small muted">(${f.acmg.points} pts)</span>
          ${C.editable ? `<button class="ghost sm" data-act="del" style="margin-left:auto">remove</button>` : ""}
        </div>
        <div style="margin:4px 0">${f.acmg.criteria.map(crit).join("")}</div>
        <div class="grid g4">
          <div><label>Section</label><select data-k="report_section" ${ro}>${C.sections.map((s) => `<option ${s === f.report_section ? "selected" : ""}>${s}</option>`).join("")}</select></div>
          <div><label>Classification override</label><select data-k="classification_override" ${ro}><option value="">(criteria-based: ${esc(f.acmg.classification)})</option>${C.classes.map((c) => `<option ${c === f.classification_override ? "selected" : ""}>${c}</option>`).join("")}</select></div>
          <div><label>Disease</label><input type="text" data-k="disease" value="${esc(f.disease)}" ${ro}></div>
          <div><label>Disease inheritance</label><input type="text" data-k="disease_inheritance" value="${esc(f.disease_inheritance)}" ${ro}></div>
        </div>
        <div style="margin-top:6px"><label>Variant interpretation (printed in report)</label><textarea data-k="interpretation" rows="4" ${ro}>${esc(f.interpretation)}</textarea></div>
        ${f.phenotype_fit && f.phenotype_fit !== "not_applicable" ? `<div class="small muted">AI phenotype fit: ${esc(f.phenotype_fit)}</div>` : ""}
      </div>`;
    }).join("");
    el.querySelectorAll(".finding").forEach((card) => {
      const f = review.findings[+card.dataset.i];
      card.querySelectorAll("[data-k]").forEach((inp) => inp.addEventListener("change", () => {
        f[inp.dataset.k] = inp.type === "checkbox" ? inp.checked : inp.value;
        if (inp.dataset.k === "include") card.classList.toggle("off", !inp.checked);
        if (inp.dataset.k === "report_section") card.className = `finding sec-${inp.value} ${f.include === false ? "off" : ""}`;
      }));
      const del = card.querySelector("[data-act=del]");
      if (del) del.addEventListener("click", () => { review.findings.splice(+card.dataset.i, 1); renderFindings(); });
    });
  }

  async function saveReview() {
    const body = {
      flag: $("rvFlag").value,
      summary_line: $("rvSummary").value,
      clinical_history_text: $("rvHistory").value,
      cnv_text: $("rvCnv") ? $("rvCnv").value : "",
      gene_list: $("rvGenes") ? $("rvGenes").value.split(/[,\s]+/).filter(Boolean) : [],
      findings: review.findings,
      clinical_interpretation: $("rvInterp").value,
      recommendations: $("rvRecs").value.split("\n"),
      limitations: $("rvLims").value.split("\n"),
    };
    setText("rvMsg", "Saving...");
    const r = await fetch(`/api/cases/${C.id}/review`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    setText("rvMsg", r.ok ? `Saved ${new Date().toLocaleTimeString()}` : `Error: ${(await r.json()).detail}`);
    if (r.ok) {
      const approveBtn = $("approveBtn");
      if (approveBtn) { approveBtn.disabled = false; approveBtn.removeAttribute("title"); }
      poll();
    }
  }

  function addFinding(key, section) {
    const v = pool[key];
    if (!v || !review) return;
    if (review.findings.some((f) => f.variant_key === key)) { alert("Already in the report list"); return; }
    const dis = (v.gene_info && v.gene_info.diseases) || [];
    review.findings.push({
      variant_key: key, report_section: section, include: true, acmg: v.acmg,
      classification: v.acmg.classification, classification_override: "",
      criteria_added: [], criteria_removed: [], phenotype_fit: "not_applicable",
      disease: dis.slice(0, 2).map((d) => d.name).join("; ") || v.clinvar_disease || "",
      disease_inheritance: ((v.gene_info && v.gene_info.inheritance) || []).join(", "),
      interpretation: "",
    });
    renderFindings();
    const overviewTab = document.querySelector('#tabs button[data-tab="overview"]');
    if (overviewTab) overviewTab.click();
    const target = $("finalizeCard");
    if (target) target.scrollIntoView({behavior: "smooth", block: "start"});
  }

  // ---------------------------------------------------------------- candidate table
  function variantTable(list, opts = {}) {
    if (!list || !list.length) return '<p class="muted small">None.</p>';
    return `<table class="list"><tr>
      ${opts.rank ? "<th>#</th>" : ""}<th class="var-col">Variant</th><th>Zygosity</th><th>Model / segregation</th><th>gnomAD</th><th>ClinVar</th>
      <th>In silico</th><th>Pheno</th><th>ACMG</th>${C.editable ? "<th></th>" : ""}</tr>
      ${list.map((v) => `<tr>
        ${opts.rank ? `<td>${v.rank ?? ""}</td>` : ""}
        <td class="var-col">${varLabel(v)}<div class="small muted">${esc(v.consequence_str)} ${v.exon ? "exon " + esc(v.exon) : ""} &middot; chr${esc(v.chrom)}:${v.pos} ${esc(v.ref)}&gt;${esc(v.alt)}</div>
          <div class="small muted">${esc(((v.gene_info && v.gene_info.diseases) || []).slice(0, 2).map((d) => d.name + (d.inheritance.length ? ` (${d.inheritance.join("/")})` : "")).join("; "))}</div></td>
        <td class="small">${zyg(v)}</td>
        <td class="small">${esc((v.models || []).join(", "))}${v.inheritance_match === false ? ' <span class="cls-P" title="Gene inheritance does not fit">!</span>' : ""}
          ${(v.model_notes || []).map((n) => `<div class="muted">${esc(n)}</div>`).join("")}</td>
        <td class="small">${fmtAf(v.pop_af)}</td>
        <td class="small">${v.clinvar_sig ? esc(v.clinvar_sig) + ` (${v.clinvar_stars}*)` : "-"}</td>
        <td class="small">${v.revel != null ? "REVEL " + v.revel.toFixed(2) + "<br>" : ""}${v.am_score != null ? "AM " + v.am_score.toFixed(2) + "<br>" : ""}${v.spliceai != null ? "SpliceAI " + v.spliceai.toFixed(2) : ""}</td>
        <td class="small">${v.phenotype_score != null ? v.phenotype_score.toFixed(2) : "-"}</td>
        <td><span class="cls-${v.acmg.short}">${v.acmg.short}</span> <span class="small muted">${v.acmg.points}</span><div>${v.acmg.criteria.map(crit).join("")}</div></td>
        ${C.editable ? `<td><select data-add="${esc(v.key)}" class="small"><option value="">+ report</option>${C.sections.filter((s) => s !== "not_reported").map((s) => `<option>${s}</option>`).join("")}</select></td>` : ""}
      </tr>`).join("")}</table>`;
  }
  function wireAdd(el) {
    if (!el) return;
    el.querySelectorAll("[data-add]").forEach((s) => s.addEventListener("change", () => { if (s.value) addFinding(s.dataset.add, s.value); s.value = ""; }));
  }

  function renderQc() {
    if (!$("qc")) return;
    const q = data.qc;
    if (!q) { setHTML("qc", '<p class="muted">No QC yet.</p>'); return; }
    const rows = Object.entries(q.samples).map(([r, s]) => `<tr><td>${esc(r)}</td><td>${s.variants_pass}</td><td>${s.snv}/${s.indel}</td><td>${s.ti_tv ?? "-"}</td><td>${s.het_hom_ratio ?? "-"}</td>
      <td>${esc(s.inferred_sex ?? "-")} <span class="muted">(declared ${esc(s.declared_sex ?? "-")}, X het ${s.x_het_ratio ?? "-"})</span></td><td>${s.roh_total_mb}</td><td>${s.low_quality_genotypes}</td></tr>`).join("");
    const roh = Object.entries(q.samples).flatMap(([r, s]) => (s.roh_segments || []).slice(0, 10).map((g) => `<tr><td>${esc(r)}</td><td>chr${esc(g.chrom)}:${g.start.toLocaleString()}-${g.end.toLocaleString()}</td><td>${((g.end - g.start) / 1e6).toFixed(1)} Mb</td><td>${g.sites}</td></tr>`)).join("");
    setHTML("qc", `${(q.warnings || []).map((w) => `<div class="warn">${esc(w)}</div>`).join("")}
      <table class="list"><tr><th>Sample</th><th>PASS variants</th><th>SNV/Indel</th><th>Ti/Tv</th><th>Het/Hom</th><th>Genetic sex</th><th>ROH Mb</th><th>Low-quality GT</th></tr>${rows}</table>
      ${q.mendelian ? `<p class="small">Mendelian errors: ${q.mendelian.errors}/${q.mendelian.checked} (${q.mendelian.error_rate != null ? (q.mendelian.error_rate * 100).toFixed(2) + "%" : "-"}) &middot; parentage ${q.parentage_confirmed ? "consistent" : "NOT confirmed"}</p>` : ""}
      ${q.couple_sharing ? `<p class="small">Shared variant fraction between partners: ${q.couple_sharing.shared_fraction ?? "-"} (higher values suggest biological relatedness)</p>` : ""}
      ${roh ? `<h3>Largest runs of homozygosity</h3><table class="list"><tr><th>Sample</th><th>Region</th><th>Length</th><th>Sites</th></tr>${roh}</table>` : ""}`);
  }

  function renderSecondary() {
    if (!$("secondary")) return;
    let html = "";
    for (const [role, vs] of Object.entries(data.secondary || {})) html += `<h3>ACMG SF v3.2 - ${esc(role)}</h3>${variantTable(vs)}`;
    if (data.carrier) html += `<h3>Carrier findings</h3>${variantTable(data.carrier)}`;
    setHTML("secondary", html || '<p class="muted">Secondary findings / carrier analysis not requested.</p>');
    wireAdd($("secondary"));
  }

  function renderCouple() {
    if (!$("couple")) return;
    const c = data.couple;
    if (!c) { setHTML("couple", ""); return; }
    const risk = c.at_risk.length ? `<table class="list"><tr><th>Gene</th><th>Mode</th><th>Partner 1</th><th>Partner 2</th><th>Risk</th></tr>${c.at_risk.map((r) => `<tr><td><i>${esc(r.gene)}</i></td><td>${esc(r.mode)}</td><td class="mono small">${r.partner1.map((k) => esc((pool[k] || {}).hgvsc || k)).join("<br>") || "-"}</td><td class="mono small">${r.partner2.map((k) => esc((pool[k] || {}).hgvsc || k)).join("<br>") || "-"}</td><td class="small">${esc(r.risk)}</td></tr>`).join("")}</table>` : '<p class="muted">No shared-gene or X-linked reproductive risk identified.</p>';
    setHTML("couple", `<h3>Reproductive risk</h3>${risk}<h3>Partner 1 carrier variants</h3>${variantTable(c.partner1)}<h3>Partner 2 carrier variants</h3>${variantTable(c.partner2)}`);
    wireAdd($("couple"));
  }

  function renderAi() {
    if (!$("aiOut")) return;
    const a = data.ai_result, p = data.phenotype_ai;
    let html = "";
    if (p) html += `<h3>Phenotype extraction</h3><p class="small">${esc(p.summary)}</p><p class="small">Suspected: ${esc((p.suspected_conditions || []).join("; ") || "-")} &middot; genes: ${esc((p.suspected_genes || []).join(", ") || "-")}</p>`;
    if (a) html += `<h3>Interpretation</h3><p><span class="flag flag-${esc(a.result_flag)}">${esc(a.result_flag)}</span> ${esc(a.flag_rationale)}</p>
      <p class="small"><b>Dismissed candidates:</b> ${esc((a.excluded_candidates || []).map((k) => (pool[k] ? pool[k].gene + " " + (pool[k].hgvsc || "") : k)).join("; ") || "-")}</p>`;
    setHTML("aiOut", html || '<p class="muted">AI was not run for this case.</p>');
  }

  async function load() {
    const r = await fetch(`/api/cases/${C.id}/data`);
    if (!r.ok) return;
    data = await r.json();
    pool = {};
    const addAll = (vs) => (vs || []).forEach((v) => { pool[v.key] = pool[v.key] || v; });
    addAll(data.candidates);
    Object.values(data.secondary || {}).forEach(addAll);
    addAll(data.carrier);
    if (data.couple) { addAll(data.couple.partner1); addAll(data.couple.partner2); }
    review = data.review ? JSON.parse(JSON.stringify(data.review)) : null;
    setHTML("warnings", (data.warnings || []).map((w) => `<div class="warn">${esc(w)}</div>`).join(""));
    renderReview();
    setText("nCand", data.candidates.length || "");
    setHTML("candidates", variantTable(data.candidates, {rank: true}));
    wireAdd($("candidates"));
    renderQc(); renderSecondary(); renderCouple(); renderAi();
  }
  load();
})();
