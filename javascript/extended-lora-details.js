/*
 * Extended LoRA Details — clipboard/prompt helpers for the details panel.
 *
 * Everything here degrades quietly: if the host page changes shape, the buttons
 * simply do nothing rather than throwing inside a Gradio event handler.
 */

function eldPromptTextarea(tabname, negative) {
    const suffix = negative ? "_neg_prompt" : "_prompt";
    const root = typeof gradioApp === "function" ? gradioApp() : document;
    if (!root) return null;
    return root.querySelector("#" + tabname + suffix + " > label > textarea");
}

function eldSeparator() {
    try {
        if (typeof opts !== "undefined" && opts && opts.extra_networks_add_text_separator) {
            return opts.extra_networks_add_text_separator;
        }
    } catch (e) {
        /* opts is not published yet */
    }
    return ", ";
}

function eldWriteToPrompt(tabname, text, mode, negative) {
    const textarea = eldPromptTextarea(tabname, negative);
    if (!textarea) return;

    const value = (text || "").trim();
    if (!value) return;

    if (mode === "replace") {
        textarea.value = value;
    } else {
        const existing = textarea.value.trim();
        textarea.value = existing ? existing + eldSeparator() + value : value;
    }

    if (typeof updateInput === "function") {
        updateInput(textarea);
    } else {
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
    }
}

/* Append the selected prompt to the tab's positive prompt box. */
function eldAppendPrompt(tabname, text) {
    eldWriteToPrompt(tabname, text, "append", false);
    return [];
}

/* Overwrite the tab's positive prompt with the selected prompt. */
function eldReplacePrompt(tabname, text) {
    eldWriteToPrompt(tabname, text, "replace", false);
    if (typeof closePopup === "function") closePopup();
    return [];
}

/*
 * UI Preset → network browsers.
 *
 * Forge renders each extra-network pane server-side, so switching the UI Preset
 * leaves the cards, the tree and the folder buttons showing the previous
 * preset's folders until the pane is rebuilt. The hidden per-tab refresh button
 * is the only path to that rebuild, so it is clicked here — the same thing the
 * pane's own ↻ control does.
 */

const eldNetworkTabs = ["txt2img", "img2img"];
let eldPresetLast = null;
let eldPresetTimer = null;

function eldGradioRoot() {
    return typeof gradioApp === "function" ? gradioApp() : document;
}

function eldOption(name, fallback) {
    try {
        if (typeof opts !== "undefined" && opts && name in opts) return opts[name];
    } catch (e) {
        /* options are not published yet */
    }
    return fallback;
}

function eldPresetValue() {
    const root = eldGradioRoot();
    if (!root) return null;
    const holder = root.querySelector("#forge_ui_preset");
    if (!holder) return null;
    const field = holder.querySelector("input, select, textarea");
    if (!field) return null;
    return (field.value || "").trim();
}

/* Rebuild every network pane. Safe to call when the panes do not exist yet. */
function eldRefreshExtraNetworks() {
    const root = eldGradioRoot();
    if (!root) return [];

    eldNetworkTabs.forEach(function (tabname) {
        // One click rebuilds every page of that tab, so the first button wins.
        const button = root.querySelector('[id^="' + tabname + '_"][id$="_extra_refresh_internal"]');
        if (button) button.dispatchEvent(new Event("click"));
    });
    return [];
}

function eldWatchPreset() {
    if (eldOption("eld_folder_filter_enabled", true) === false) return;
    if (eldOption("eld_folder_filter_auto_refresh", true) === false) return;

    const value = eldPresetValue();
    if (value === null || value === "") return;

    if (eldPresetLast === null) {
        eldPresetLast = value; // first sighting: the panes already match it
        return;
    }
    if (value === eldPresetLast) return;

    eldPresetLast = value;
    // The dropdown's text changes while it is being typed in, so settle first
    // and rebuild once.
    clearTimeout(eldPresetTimer);
    eldPresetTimer = setTimeout(eldRefreshExtraNetworks, 600);
}

if (typeof onUiLoaded === "function") {
    onUiLoaded(function () {
        eldPresetLast = eldPresetValue();
    });
}

if (typeof onAfterUiUpdate === "function") {
    onAfterUiUpdate(eldWatchPreset);
}
