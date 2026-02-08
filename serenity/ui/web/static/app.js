/* ═══════════════════════════════════════════════════════════
   Serenity Web UI — Application Logic
   ═══════════════════════════════════════════════════════════ */
"use strict";

/* ─────── Global State ─────── */
const state = {
    models: { checkpoints: [], loras: [], vaes: [] },
    selectedModel: null,
    activeLoras: [],          // [{path, name, weight}]
    generating: false,
    currentSeed: -1,
    lastSeed: -1,
    history: [],              // [{url, params, timestamp}]
    currentImage: null,       // {url, params}
    batchImages: [],
    ws: null,
    wsReconnectTimer: null,
    filterText: "",
};

/* ═══════════════════════════════════════════════════════════
   Initialization
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
    initSliders();
    initPromptAreas();
    initButtons();
    initBottomTabs();
    initTopbarTabs();
    initKeyboardShortcuts();
    initLightbox();
    initRightPanel();
    connectWebSocket();
    fetchModels();
    fetchSamplers();
    fetchSchedulers();
    fetchHistory();
});

/* ═══════════════════════════════════════════════════════════
   Slider / Input Binding
   ═══════════════════════════════════════════════════════════ */

function initSliders() {
    bindSlider("steps-slider", "steps-value", v => v);
    bindSlider("cfg-slider", "cfg-value", v => parseFloat(v).toFixed(1));
    bindSlider("batch-slider", "batch-value", v => v);
    bindSlider("rescale-slider", "rescale-value", v => parseFloat(v).toFixed(2));
    bindSlider("clip-skip-slider", "clip-skip-value", v => v);
}

function bindSlider(sliderId, valueId, formatter) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(valueId);
    if (!slider || !display) return;
    const update = () => { display.textContent = formatter(slider.value); };
    slider.addEventListener("input", update);
    update();
}

/* ═══════════════════════════════════════════════════════════
   Prompt Areas
   ═══════════════════════════════════════════════════════════ */

function initPromptAreas() {
    const prompt = document.getElementById("prompt");
    const negPrompt = document.getElementById("negative-prompt");
    const promptChars = document.getElementById("prompt-chars");
    const negChars = document.getElementById("neg-chars");

    prompt.addEventListener("input", () => { promptChars.textContent = prompt.value.length; });
    negPrompt.addEventListener("input", () => { negChars.textContent = negPrompt.value.length; });

    // Ctrl+Enter to generate from prompt
    prompt.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); generate(); }
    });
    negPrompt.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); generate(); }
    });
}

/* ═══════════════════════════════════════════════════════════
   Buttons
   ═══════════════════════════════════════════════════════════ */

function initButtons() {
    document.getElementById("generate-btn").addEventListener("click", generate);
    document.getElementById("interrupt-btn").addEventListener("click", interrupt);

    // Seed buttons
    document.getElementById("seed-random").addEventListener("click", () => {
        document.getElementById("seed-input").value = Math.floor(Math.random() * 4294967296);
    });
    document.getElementById("seed-reuse").addEventListener("click", () => {
        if (state.lastSeed >= 0) {
            document.getElementById("seed-input").value = state.lastSeed;
        }
    });

    // Swap dimensions
    document.getElementById("swap-dims").addEventListener("click", () => {
        const w = document.getElementById("width-input");
        const h = document.getElementById("height-input");
        const tmp = w.value;
        w.value = h.value;
        h.value = tmp;
        updateAspectPresetHighlight();
    });

    // Aspect presets
    document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.getElementById("width-input").value = btn.dataset.w;
            document.getElementById("height-input").value = btn.dataset.h;
            updateAspectPresetHighlight();
        });
    });

    // Add LoRA button -> switch to LoRAs tab
    document.getElementById("add-lora-btn").addEventListener("click", () => {
        switchBottomTab("loras-browse");
    });

    // Reuse / Copy params
    document.getElementById("reuse-params-btn").addEventListener("click", () => {
        if (state.currentImage) reuseParams(state.currentImage.params);
    });
    document.getElementById("copy-params-btn").addEventListener("click", () => {
        if (state.currentImage) {
            const text = JSON.stringify(state.currentImage.params, null, 2);
            navigator.clipboard.writeText(text).then(() => showToast("Parameters copied", "success"));
        }
    });
}

function updateAspectPresetHighlight() {
    const w = parseInt(document.getElementById("width-input").value);
    const h = parseInt(document.getElementById("height-input").value);
    document.querySelectorAll(".preset-btn").forEach(btn => {
        const match = parseInt(btn.dataset.w) === w && parseInt(btn.dataset.h) === h;
        btn.classList.toggle("active", match);
    });
}

/* ═══════════════════════════════════════════════════════════
   Collapsible Sections
   ═══════════════════════════════════════════════════════════ */

function toggleSection(sectionId) {
    const body = document.getElementById("body-" + sectionId);
    const toggle = body.previousElementSibling;
    const icon = toggle.querySelector(".toggle-icon");
    const isOpen = body.classList.toggle("open");
    toggle.setAttribute("aria-expanded", isOpen);
    icon.innerHTML = isOpen ? "&#9660;" : "&#9654;";
}

/* ═══════════════════════════════════════════════════════════
   Bottom Tabs
   ═══════════════════════════════════════════════════════════ */

function initBottomTabs() {
    document.querySelectorAll(".bottom-tab").forEach(tab => {
        tab.addEventListener("click", () => switchBottomTab(tab.dataset.btab));
    });

    // Search / filter
    document.getElementById("bottom-search").addEventListener("input", (e) => {
        state.filterText = e.target.value.toLowerCase();
        applyBottomFilter();
    });
}

function switchBottomTab(tabName) {
    document.querySelectorAll(".bottom-tab").forEach(t => t.classList.toggle("active", t.dataset.btab === tabName));
    document.querySelectorAll(".bottom-pane").forEach(p => p.classList.toggle("active", p.id === "pane-" + tabName));
}

function applyBottomFilter() {
    // Filter model cards
    document.querySelectorAll("#models-grid .model-card").forEach(card => {
        const name = card.querySelector(".model-card-name").textContent.toLowerCase();
        card.style.display = name.includes(state.filterText) ? "" : "none";
    });
    // Filter LoRA cards
    document.querySelectorAll("#loras-grid .model-card").forEach(card => {
        const name = card.querySelector(".model-card-name").textContent.toLowerCase();
        card.style.display = name.includes(state.filterText) ? "" : "none";
    });
    // Filter VAE cards
    document.querySelectorAll("#vaes-grid .model-card").forEach(card => {
        const name = card.querySelector(".model-card-name").textContent.toLowerCase();
        card.style.display = name.includes(state.filterText) ? "" : "none";
    });
}

/* ═══════════════════════════════════════════════════════════
   Top Bar Tabs
   ═══════════════════════════════════════════════════════════ */

function initTopbarTabs() {
    document.querySelectorAll(".topbar-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".topbar-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            // Future: switch modes (simple vs. advanced)
        });
    });
}

/* ═══════════════════════════════════════════════════════════
   Keyboard Shortcuts
   ═══════════════════════════════════════════════════════════ */

function initKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
        // Ctrl+Enter to generate (global, unless in textarea already handled)
        if (e.ctrlKey && e.key === "Enter" && document.activeElement.tagName !== "TEXTAREA") {
            e.preventDefault();
            generate();
        }
        // Escape to interrupt or close lightbox
        if (e.key === "Escape") {
            if (!document.getElementById("lightbox").classList.contains("hidden")) {
                closeLightbox();
            } else if (state.generating) {
                interrupt();
            }
        }
    });
}

/* ═══════════════════════════════════════════════════════════
   Lightbox
   ═══════════════════════════════════════════════════════════ */

function initLightbox() {
    const lb = document.getElementById("lightbox");
    lb.addEventListener("click", (e) => {
        if (e.target === lb) closeLightbox();
    });
    document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
    document.getElementById("main-image").addEventListener("click", () => {
        if (state.currentImage) openLightbox(state.currentImage.url);
    });
}

function openLightbox(src) {
    const lb = document.getElementById("lightbox");
    document.getElementById("lightbox-img").src = src;
    lb.classList.remove("hidden");
    lb.focus();
}

function closeLightbox() {
    document.getElementById("lightbox").classList.add("hidden");
}

/* ═══════════════════════════════════════════════════════════
   Right Panel
   ═══════════════════════════════════════════════════════════ */

function initRightPanel() {
    document.getElementById("collapse-right").addEventListener("click", () => {
        document.getElementById("right-panel").classList.toggle("collapsed");
    });
}

/* ═══════════════════════════════════════════════════════════
   WebSocket
   ═══════════════════════════════════════════════════════════ */

function connectWebSocket() {
    if (state.ws && state.ws.readyState <= 1) return;

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    try {
        state.ws = new WebSocket(`${protocol}//${location.host}/ws`);
    } catch {
        setWSStatus(false);
        scheduleReconnect();
        return;
    }

    state.ws.onopen = () => {
        setWSStatus(true);
        if (state.wsReconnectTimer) {
            clearTimeout(state.wsReconnectTimer);
            state.wsReconnectTimer = null;
        }
    };

    state.ws.onmessage = (event) => {
        let msg;
        try {
            msg = JSON.parse(event.data);
        } catch { return; }

        switch (msg.type) {
            case "progress":
                updateProgress(msg);
                break;
            case "complete":
                onGenerationComplete(msg);
                break;
            case "error":
                onGenerationError(msg);
                break;
            case "vram":
                updateVRAM(msg);
                break;
        }
    };

    state.ws.onclose = () => {
        setWSStatus(false);
        scheduleReconnect();
    };

    state.ws.onerror = () => {
        setWSStatus(false);
    };
}

function scheduleReconnect() {
    if (!state.wsReconnectTimer) {
        state.wsReconnectTimer = setTimeout(() => {
            state.wsReconnectTimer = null;
            connectWebSocket();
        }, 3000);
    }
}

function setWSStatus(connected) {
    const el = document.getElementById("ws-status");
    el.classList.toggle("connected", connected);
    el.classList.toggle("disconnected", !connected);
    el.title = connected ? "WebSocket connected" : "WebSocket disconnected";
    el.querySelector(".ws-label").textContent = connected ? "Connected" : "Disconnected";
}

/* ═══════════════════════════════════════════════════════════
   VRAM
   ═══════════════════════════════════════════════════════════ */

function updateVRAM(msg) {
    const pct = msg.percent || 0;
    const text = msg.text || `${pct}%`;
    document.getElementById("vram-fill").style.width = pct + "%";
    document.getElementById("vram-text").textContent = text;
    const fill = document.getElementById("vram-fill");
    fill.style.background = pct > 90 ? "var(--error)" : pct > 70 ? "var(--warning)" : "var(--accent)";
}

/* ═══════════════════════════════════════════════════════════
   API Calls
   ═══════════════════════════════════════════════════════════ */

async function apiFetch(url, options = {}) {
    try {
        const res = await fetch(url, options);
        if (!res.ok) {
            const text = await res.text().catch(() => res.statusText);
            throw new Error(`${res.status}: ${text}`);
        }
        return await res.json();
    } catch (err) {
        console.error(`API error (${url}):`, err);
        throw err;
    }
}

/* ─── Models ─── */

async function fetchModels() {
    try {
        const data = await apiFetch("/api/models");
        const ckpts = data.checkpoints || data.checkpoint || [];
        const diffModels = data.diffusion_model || data.diffusion_models || [];
        // Filter out multi-file shards (e.g. "model-00003-of-00004")
        const allModels = [...ckpts, ...diffModels].filter(m => {
            const name = m.name || m;
            return !/-\d{5}-of-\d{5}$/.test(name);
        });
        state.models.checkpoints = allModels;
        state.models.loras = data.loras || data.lora || [];
        state.models.vaes = data.vaes || data.vae || [];
        populateModelDropdown(state.models.checkpoints);
        populateModelsGrid(state.models.checkpoints);
        populateLorasGrid(state.models.loras);
        populateVAEsGrid(state.models.vaes);
    } catch {
        showToast("Failed to load models. Is the server running?", "error");
        // Populate with placeholder so UI is usable
        populateModelDropdown([]);
    }
}

function populateModelDropdown(models) {
    const select = document.getElementById("model-select");
    select.innerHTML = "";
    if (models.length === 0) {
        select.innerHTML = '<option value="">No models found</option>';
        return;
    }
    models.forEach(m => {
        const opt = document.createElement("option");
        const name = m.name || m;
        const path = m.path || m;
        opt.value = path;
        opt.textContent = name;
        select.appendChild(opt);
    });
    state.selectedModel = select.value;
    select.addEventListener("change", () => { state.selectedModel = select.value; });
}

function populateModelsGrid(models) {
    const grid = document.getElementById("models-grid");
    grid.innerHTML = "";
    if (models.length === 0) {
        grid.innerHTML = '<div class="thumb-placeholder">No models found</div>';
        return;
    }
    models.forEach(m => {
        const name = m.name || m;
        const path = m.path || m;
        const size = m.size_mb ? `${(m.size_mb / 1024).toFixed(1)} GB` : "";
        const card = document.createElement("div");
        card.className = "model-card";
        card.innerHTML = `
            <span class="model-card-name" title="${escapeHtml(path)}">${escapeHtml(name)}</span>
            ${size ? `<span class="model-card-meta">${size}</span>` : ""}
        `;
        card.addEventListener("click", () => {
            document.getElementById("model-select").value = path;
            state.selectedModel = path;
            grid.querySelectorAll(".model-card").forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            showToast(`Model: ${name}`, "info");
        });
        grid.appendChild(card);
    });
}

function populateLorasGrid(loras) {
    const grid = document.getElementById("loras-grid");
    grid.innerHTML = "";
    if (loras.length === 0) {
        grid.innerHTML = '<div class="thumb-placeholder">No LoRAs found</div>';
        return;
    }
    loras.forEach(m => {
        const name = m.name || m;
        const path = m.path || m;
        const size = m.size_mb ? `${m.size_mb.toFixed(0)} MB` : "";
        const card = document.createElement("div");
        card.className = "model-card";
        card.innerHTML = `
            <span class="model-card-name" title="${escapeHtml(path)}">${escapeHtml(name)}</span>
            ${size ? `<span class="model-card-meta">${size}</span>` : ""}
        `;
        card.addEventListener("click", () => addLora(path, name));
        grid.appendChild(card);
    });
}

function populateVAEsGrid(vaes) {
    const grid = document.getElementById("vaes-grid");
    grid.innerHTML = "";
    if (vaes.length === 0) {
        grid.innerHTML = '<div class="thumb-placeholder">No VAEs found</div>';
        return;
    }
    vaes.forEach(m => {
        const name = m.name || m;
        const path = m.path || m;
        const size = m.size_mb ? `${m.size_mb.toFixed(0)} MB` : "";
        const card = document.createElement("div");
        card.className = "model-card";
        card.innerHTML = `
            <span class="model-card-name" title="${escapeHtml(path)}">${escapeHtml(name)}</span>
            ${size ? `<span class="model-card-meta">${size}</span>` : ""}
        `;
        grid.appendChild(card);
    });
}

/* ─── Samplers / Schedulers ─── */

async function fetchSamplers() {
    try {
        const data = await apiFetch("/api/samplers");
        const select = document.getElementById("sampler-select");
        if (Array.isArray(data) && data.length > 0) {
            select.innerHTML = "";
            data.forEach(s => {
                const opt = document.createElement("option");
                const name = typeof s === "string" ? s : s.name || s;
                opt.value = name;
                opt.textContent = name;
                select.appendChild(opt);
            });
        }
    } catch {
        // Keep defaults in HTML
    }
}

async function fetchSchedulers() {
    try {
        const data = await apiFetch("/api/schedulers");
        const select = document.getElementById("scheduler-select");
        if (Array.isArray(data) && data.length > 0) {
            select.innerHTML = "";
            data.forEach(s => {
                const opt = document.createElement("option");
                const name = typeof s === "string" ? s : s.name || s;
                opt.value = name;
                opt.textContent = name;
                select.appendChild(opt);
            });
        }
    } catch {
        // Keep defaults in HTML
    }
}

/* ─── History ─── */

async function fetchHistory() {
    try {
        const data = await apiFetch("/api/history");
        if (Array.isArray(data)) {
            state.history = data;
            renderHistoryGrid();
        }
    } catch {
        // History not available, that's fine
    }
}

function renderHistoryGrid() {
    const grid = document.getElementById("history-grid");
    grid.innerHTML = "";
    if (state.history.length === 0) {
        grid.innerHTML = '<div class="thumb-placeholder">No images yet. Generate one!</div>';
        return;
    }
    state.history.forEach((item, idx) => {
        const url = item.url || item;
        const card = document.createElement("div");
        card.className = "thumb-card";
        card.innerHTML = `<img src="${escapeHtml(url)}" alt="History ${idx}" loading="lazy">`;
        card.addEventListener("click", () => {
            displayImage(url, item.params || null);
        });
        grid.appendChild(card);
    });
}

/* ═══════════════════════════════════════════════════════════
   LoRA Management
   ═══════════════════════════════════════════════════════════ */

function addLora(path, name) {
    // Don't add duplicates
    if (state.activeLoras.some(l => l.path === path)) {
        showToast("LoRA already added", "warning");
        return;
    }
    state.activeLoras.push({ path, name: name || path.split("/").pop(), weight: 1.0 });
    renderActiveLoras();
    updateLoraCount();
    // Open the LoRA section
    const body = document.getElementById("body-loras");
    if (!body.classList.contains("open")) toggleSection("loras");
    showToast(`Added LoRA: ${name || path}`, "info");
}

function removeLora(index) {
    state.activeLoras.splice(index, 1);
    renderActiveLoras();
    updateLoraCount();
}

function renderActiveLoras() {
    const container = document.getElementById("active-loras");
    container.innerHTML = "";
    state.activeLoras.forEach((lora, idx) => {
        const item = document.createElement("div");
        item.className = "lora-item";
        item.innerHTML = `
            <span class="lora-name" title="${escapeHtml(lora.path)}">${escapeHtml(lora.name)}</span>
            <input type="range" class="param-slider lora-weight-slider" min="0" max="2" step="0.05" value="${lora.weight}">
            <span class="lora-weight">${lora.weight.toFixed(2)}</span>
            <button class="lora-remove" title="Remove">&times;</button>
        `;
        const slider = item.querySelector(".lora-weight-slider");
        const display = item.querySelector(".lora-weight");
        slider.addEventListener("input", () => {
            lora.weight = parseFloat(slider.value);
            display.textContent = lora.weight.toFixed(2);
        });
        item.querySelector(".lora-remove").addEventListener("click", () => removeLora(idx));
        container.appendChild(item);
    });
}

function updateLoraCount() {
    const el = document.getElementById("lora-count");
    el.textContent = state.activeLoras.length > 0 ? state.activeLoras.length : "";
}

/* ═══════════════════════════════════════════════════════════
   Generation
   ═══════════════════════════════════════════════════════════ */

function gatherParams() {
    const seedVal = parseInt(document.getElementById("seed-input").value);
    return {
        model: document.getElementById("model-select").value,
        prompt: document.getElementById("prompt").value,
        negative_prompt: document.getElementById("negative-prompt").value,
        seed: seedVal === -1 ? Math.floor(Math.random() * 4294967296) : seedVal,
        steps: parseInt(document.getElementById("steps-slider").value),
        cfg_scale: parseFloat(document.getElementById("cfg-slider").value),
        width: parseInt(document.getElementById("width-input").value),
        height: parseInt(document.getElementById("height-input").value),
        sampler: document.getElementById("sampler-select").value,
        scheduler: document.getElementById("scheduler-select").value,
        batch_size: parseInt(document.getElementById("batch-slider").value),
        rescale_cfg: parseFloat(document.getElementById("rescale-slider").value),
        clip_skip: parseInt(document.getElementById("clip-skip-slider").value),
        mahiro: document.getElementById("mahiro-toggle").checked,
        loras: state.activeLoras.map(l => ({ path: l.path, weight: l.weight })),
    };
}

async function generate() {
    if (state.generating) return;

    const params = gatherParams();
    if (!params.prompt.trim()) {
        showToast("Enter a prompt first", "warning");
        document.getElementById("prompt").focus();
        return;
    }

    state.generating = true;
    state.currentSeed = params.seed;
    setGeneratingUI(true);

    try {
        const data = await apiFetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });

        // If server responds immediately with result (non-WebSocket flow)
        if (data && data.images) {
            onGenerationComplete({ images: data.images, params, seed: params.seed, time: data.time });
        }
        // Otherwise, progress comes via WebSocket
    } catch (err) {
        onGenerationError({ message: err.message || "Generation failed" });
    }
}

async function interrupt() {
    try {
        await apiFetch("/api/interrupt", { method: "POST" });
        showToast("Interrupted", "warning");
    } catch {
        showToast("Failed to interrupt", "error");
    }
}

function setGeneratingUI(isGenerating) {
    const genBtn = document.getElementById("generate-btn");
    const intBtn = document.getElementById("interrupt-btn");
    const overlay = document.getElementById("progress-overlay");

    genBtn.disabled = isGenerating;
    intBtn.classList.toggle("hidden", !isGenerating);
    overlay.classList.toggle("hidden", !isGenerating);

    if (isGenerating) {
        document.getElementById("progress-bar").style.width = "0%";
        document.getElementById("progress-steps").textContent = "0 / ?";
        document.getElementById("progress-text").textContent = "Generating...";
    }
}

/* ─── Progress ─── */

function updateProgress(msg) {
    const step = msg.step || 0;
    const total = msg.total || 1;
    const pct = Math.round((step / total) * 100);
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("progress-steps").textContent = `${step} / ${total}`;
    if (msg.text) document.getElementById("progress-text").textContent = msg.text;
}

/* ─── Complete ─── */

function onGenerationComplete(msg) {
    state.generating = false;
    setGeneratingUI(false);

    const images = msg.images || [];
    const params = msg.params || gatherParams();
    const seed = msg.seed || params.seed;
    const time = msg.time || null;
    params.seed = seed;
    state.lastSeed = seed;

    if (images.length === 0) {
        showToast("No images returned", "warning");
        return;
    }

    // Display first image
    const mainUrl = typeof images[0] === "string" ? images[0] : images[0].url;
    displayImage(mainUrl, { ...params, time });

    // Batch thumbnails
    state.batchImages = images.map(i => typeof i === "string" ? i : i.url);
    renderBatchThumbs();

    // Add to history
    images.forEach(img => {
        const url = typeof img === "string" ? img : img.url;
        state.history.unshift({ url, params: { ...params, time }, timestamp: Date.now() });
    });
    renderHistoryGrid();

    showToast(`Generated in ${time ? time.toFixed(1) + "s" : "?"}`, "success");
}

/* ─── Error ─── */

function onGenerationError(msg) {
    state.generating = false;
    setGeneratingUI(false);
    const message = msg.message || msg.error || "Unknown error";
    showToast(`Generation error: ${message}`, "error");
}

/* ═══════════════════════════════════════════════════════════
   Image Display
   ═══════════════════════════════════════════════════════════ */

function displayImage(url, params) {
    const mainImg = document.getElementById("main-image");
    const placeholder = document.getElementById("image-placeholder");

    mainImg.src = url;
    mainImg.classList.remove("hidden");
    placeholder.classList.add("hidden");

    state.currentImage = { url, params };

    // Update metadata overlay
    updateMetaOverlay(params);

    // Update right panel
    updateGenInfo(params);
}

function updateMetaOverlay(params) {
    const overlay = document.getElementById("meta-overlay");
    const content = document.getElementById("meta-overlay-content");
    if (!params) { overlay.classList.add("hidden"); return; }

    overlay.classList.remove("hidden");
    const tags = [];
    if (params.model) tags.push(shortName(params.model));
    if (params.seed != null) tags.push(`Seed: ${params.seed}`);
    if (params.steps) tags.push(`${params.steps} steps`);
    if (params.cfg_scale != null) tags.push(`CFG ${params.cfg_scale}`);
    if (params.width && params.height) tags.push(`${params.width}x${params.height}`);
    if (params.sampler) tags.push(params.sampler);
    if (params.time) tags.push(`${params.time.toFixed(1)}s`);

    content.innerHTML = tags.map(t => `<span class="meta-tag">${escapeHtml(t)}</span>`).join("");
}

function updateGenInfo(params) {
    const info = document.getElementById("gen-info");
    const actions = document.getElementById("gen-actions");
    const raw = document.getElementById("raw-params");

    if (!params) {
        info.innerHTML = '<p class="gen-info-empty">No generation data yet.</p>';
        actions.style.display = "none";
        raw.innerHTML = "";
        return;
    }

    const rows = [
        ["Model", shortName(params.model)],
        ["Seed", params.seed],
        ["Steps", params.steps],
        ["CFG Scale", params.cfg_scale],
        ["Size", `${params.width} x ${params.height}`],
        ["Sampler", params.sampler],
        ["Scheduler", params.scheduler],
        ["Batch", params.batch_size],
    ];
    if (params.time) rows.push(["Time", `${params.time.toFixed(1)}s`]);
    if (params.clip_skip) rows.push(["CLIP Skip", params.clip_skip]);
    if (params.rescale_cfg) rows.push(["Rescale CFG", params.rescale_cfg]);
    if (params.mahiro) rows.push(["MaHiRo", "On"]);
    if (params.loras && params.loras.length) {
        rows.push(["LoRAs", params.loras.map(l => `${shortName(l.path)} (${l.weight})`).join(", ")]);
    }

    info.innerHTML = rows.map(([label, value]) =>
        `<div class="gen-info-row">
            <span class="gen-info-label">${escapeHtml(label)}</span>
            <span class="gen-info-value">${escapeHtml(String(value || ""))}</span>
        </div>`
    ).join("");

    // Prompt display
    if (params.prompt) {
        info.innerHTML += `
            <div style="margin-top:8px">
                <div class="gen-info-label" style="margin-bottom:4px">Prompt</div>
                <div style="font-size:11px;color:var(--text-primary);word-break:break-word">${escapeHtml(params.prompt)}</div>
            </div>`;
    }
    if (params.negative_prompt) {
        info.innerHTML += `
            <div style="margin-top:6px">
                <div class="gen-info-label" style="margin-bottom:4px">Negative</div>
                <div style="font-size:11px;color:var(--text-secondary);word-break:break-word">${escapeHtml(params.negative_prompt)}</div>
            </div>`;
    }

    actions.style.display = "flex";

    // Raw JSON
    raw.innerHTML = `<pre>${escapeHtml(JSON.stringify(params, null, 2))}</pre>`;
}

/* ─── Batch Thumbnails ─── */

function renderBatchThumbs() {
    const container = document.getElementById("batch-thumbs");
    if (state.batchImages.length <= 1) {
        container.classList.add("hidden");
        return;
    }
    container.classList.remove("hidden");
    container.innerHTML = "";
    state.batchImages.forEach((url, idx) => {
        const img = document.createElement("img");
        img.className = "batch-thumb" + (idx === 0 ? " active" : "");
        img.src = url;
        img.alt = `Batch ${idx + 1}`;
        img.addEventListener("click", () => {
            container.querySelectorAll(".batch-thumb").forEach(t => t.classList.remove("active"));
            img.classList.add("active");
            const mainImg = document.getElementById("main-image");
            mainImg.src = url;
            state.currentImage = { url, params: state.currentImage?.params };
        });
        container.appendChild(img);
    });
}

/* ═══════════════════════════════════════════════════════════
   Reuse Parameters
   ═══════════════════════════════════════════════════════════ */

function reuseParams(params) {
    if (!params) return;
    if (params.model) {
        const select = document.getElementById("model-select");
        if ([...select.options].some(o => o.value === params.model)) {
            select.value = params.model;
            state.selectedModel = params.model;
        }
    }
    if (params.prompt != null) document.getElementById("prompt").value = params.prompt;
    if (params.negative_prompt != null) document.getElementById("negative-prompt").value = params.negative_prompt;
    if (params.seed != null) document.getElementById("seed-input").value = params.seed;
    if (params.steps != null) setSlider("steps-slider", "steps-value", params.steps, v => v);
    if (params.cfg_scale != null) setSlider("cfg-slider", "cfg-value", params.cfg_scale, v => parseFloat(v).toFixed(1));
    if (params.width != null) document.getElementById("width-input").value = params.width;
    if (params.height != null) document.getElementById("height-input").value = params.height;
    if (params.sampler) document.getElementById("sampler-select").value = params.sampler;
    if (params.scheduler) document.getElementById("scheduler-select").value = params.scheduler;
    if (params.batch_size != null) setSlider("batch-slider", "batch-value", params.batch_size, v => v);
    if (params.rescale_cfg != null) setSlider("rescale-slider", "rescale-value", params.rescale_cfg, v => parseFloat(v).toFixed(2));
    if (params.clip_skip != null) setSlider("clip-skip-slider", "clip-skip-value", params.clip_skip, v => v);
    if (params.mahiro != null) document.getElementById("mahiro-toggle").checked = params.mahiro;
    if (params.loras) {
        state.activeLoras = params.loras.map(l => ({
            path: l.path,
            name: shortName(l.path),
            weight: l.weight,
        }));
        renderActiveLoras();
        updateLoraCount();
    }
    updateAspectPresetHighlight();
    // Update char counts
    document.getElementById("prompt-chars").textContent = (params.prompt || "").length;
    document.getElementById("neg-chars").textContent = (params.negative_prompt || "").length;
    showToast("Parameters loaded", "info");
}

function setSlider(sliderId, valueId, val, formatter) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(valueId);
    if (slider && display) {
        slider.value = val;
        display.textContent = formatter(val);
    }
}

/* ═══════════════════════════════════════════════════════════
   Toast Notifications
   ═══════════════════════════════════════════════════════════ */

function showToast(message, type = "info", duration = 3500) {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("fadeout");
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/* ═══════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════ */

function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

function shortName(path) {
    if (!path) return "";
    const parts = String(path).replace(/\\/g, "/").split("/");
    const name = parts[parts.length - 1];
    // Remove extension
    const dotIdx = name.lastIndexOf(".");
    return dotIdx > 0 ? name.substring(0, dotIdx) : name;
}
