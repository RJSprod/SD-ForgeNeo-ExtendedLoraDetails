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
 * The panel's own buttons: copy, the trigger-word chips, the prompt cards.
 *
 * The panel is HTML that Gradio replaces every time the dialog opens, so one
 * listener on the document handles every [data-eld-action] inside it. Each
 * section keeps the exact text its buttons act on in a hidden .eld-copy-source.
 */

function eldCopyFallback(value) {
    const area = document.createElement("textarea");
    area.value = value;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-9999px";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    let copied = false;
    try {
        copied = document.execCommand("copy");
    } catch (e) {
        copied = false;
    }
    area.remove();
    return copied;
}

/* The clipboard API needs a secure context; a LAN address over http is not one. */
function eldCopyText(text) {
    const value = String(text || "");
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(value).then(
            function () {
                return true;
            },
            function () {
                return eldCopyFallback(value);
            }
        );
    }
    return Promise.resolve(eldCopyFallback(value));
}

function eldSourceText(scope) {
    const source = scope ? scope.querySelector(".eld-copy-source") : null;
    if (!source) return "";
    return source.tagName === "TEXTAREA" ? source.value : source.textContent;
}

function eldFlash(button, label) {
    if (!button || button.dataset.eldLabel) return;
    button.dataset.eldLabel = button.textContent;
    button.textContent = label;
    setTimeout(function () {
        button.textContent = button.dataset.eldLabel;
        delete button.dataset.eldLabel;
    }, 1200);
}

function eldSplitWords(text) {
    return String(text || "")
        .split(/ *, */)
        .filter(function (word) {
            return word.trim();
        });
}

function eldActivationField(section) {
    const id = section ? section.getAttribute("data-eld-target") : "";
    if (!id) return null;
    const holder = eldGradioRoot().querySelector("#" + CSS.escape(id));
    return holder ? holder.querySelector("textarea, input") : null;
}

function eldSetField(field, value) {
    field.value = value;
    if (typeof updateInput === "function") {
        updateInput(field);
    } else {
        field.dispatchEvent(new Event("input", { bubbles: true }));
    }
}

/* Light up the chips whose word is in the activation text right now. */
function eldSyncChips(section) {
    const field = eldActivationField(section);
    if (!field) return;
    const words = eldSplitWords(field.value);
    section.querySelectorAll(".eld-word").forEach(function (chip) {
        chip.classList.toggle("eld-word-on", words.indexOf(chip.getAttribute("data-word")) !== -1);
    });
}

function eldTriggerAction(section, action, chip) {
    const field = eldActivationField(section);
    if (!field) return;
    let words = eldSplitWords(field.value);

    if (action === "toggle-trigger") {
        const word = chip.getAttribute("data-word");
        if (words.indexOf(word) !== -1) {
            words = words.filter(function (piece) {
                return piece !== word;
            });
        } else {
            words.push(word);
        }
        eldSetField(field, words.join(", "));
    } else if (action === "use-triggers") {
        eldSetField(field, eldSourceText(section));
    } else if (action === "append-triggers") {
        eldSplitWords(eldSourceText(section)).forEach(function (word) {
            if (words.indexOf(word) === -1) words.push(word);
        });
        eldSetField(field, words.join(", "));
    }
    eldSyncChips(section);
}

/* Hand a card's request to Python through the hidden textbox + button pair. */
function eldBridge(id, value) {
    const root = eldGradioRoot();
    const area = root.querySelector("#" + CSS.escape(id) + " textarea");
    const button = root.querySelector("#" + CSS.escape(id + "_button"));
    if (!area || !button) return false;
    eldSetField(area, value);
    button.click();
    return true;
}

function eldCardAction(card, action, button) {
    const grid = card.closest("[data-eld-bridge]");
    const tab = grid ? grid.getAttribute("data-eld-tab") : "txt2img";
    const index = card.getAttribute("data-index");

    if (action === "append-prompt") {
        eldAppendPrompt(tab, eldSourceText(card));
        eldFlash(button, "Appended");
    } else if (action === "replace-prompt") {
        eldReplacePrompt(tab, eldSourceText(card));
    } else if (action === "view" || action === "download") {
        if (!grid || !eldBridge(grid.getAttribute("data-eld-bridge"), action + ":" + index)) return;
        if (action === "download") {
            button.disabled = true;
            button.textContent = "Downloading…";
        } else {
            const panel = card.closest(".eld-panel");
            const viewer = panel ? panel.querySelector(".eld-picker") : null;
            if (viewer) viewer.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }
}

function eldHandleAction(button) {
    const action = button.getAttribute("data-eld-action");
    const card = button.closest(".eld-card");
    if (card && action !== "copy") {
        eldCardAction(card, action, button);
        return;
    }

    const section = button.closest(".eld-section") || card;
    if (action === "copy") {
        eldCopyText(eldSourceText(card || section)).then(function (copied) {
            eldFlash(button, copied ? "Copied" : "Copy failed");
        });
    } else if (action === "copy-word") {
        eldCopyText(button.getAttribute("data-word")).then(function (copied) {
            eldFlash(button, copied ? "Copied" : "Copy failed");
        });
    } else if (action === "toggle-trigger" || action === "use-triggers" || action === "append-triggers") {
        eldTriggerAction(section, action, button);
    } else if (action === "append-negative") {
        const holder = button.closest("[data-eld-tab]");
        eldWriteToPrompt(holder ? holder.getAttribute("data-eld-tab") : "txt2img", eldSourceText(section), "append", true);
        eldFlash(button, "Appended");
    } else if (action === "expand") {
        const open = section.classList.toggle("eld-expanded");
        button.textContent = open ? "Show less" : "Show more";
        button.setAttribute("aria-expanded", open ? "true" : "false");
    }
}

/* Typing in the activation text keeps the chips' highlighting honest. */
document.addEventListener("input", function (event) {
    const field = event.target;
    const holder = field && field.closest ? field.closest("[id]") : null;
    if (!holder || !holder.id) return;
    eldGradioRoot()
        .querySelectorAll('.eld-triggers[data-eld-target="' + CSS.escape(holder.id) + '"]')
        .forEach(eldSyncChips);
});

document.addEventListener("click", function (event) {
    const target = event.target;
    const button = target && target.closest ? target.closest("[data-eld-action]") : null;
    if (!button || !button.closest(".eld-panel")) return;
    event.preventDefault();
    try {
        eldHandleAction(button);
    } catch (e) {
        console.error("[Extended LoRA Details]", e);
    }
});

/* Python clamps a text by its length; one that turns out to fit on screen needs
   neither the fade nor the "Show more". Measured once it is visible. */
function eldFitClamps() {
    eldGradioRoot()
        .querySelectorAll(".eld-panel .eld-clamp:not([data-eld-fit])")
        .forEach(function (text) {
            if (!text.offsetParent || !text.clientHeight || text.closest(".eld-expanded")) return;
            text.setAttribute("data-eld-fit", "1");
            if (text.scrollHeight > text.clientHeight + 4) return;
            text.classList.remove("eld-clamp");
            const toggle = text.parentElement ? text.parentElement.querySelector('[data-eld-action="expand"]') : null;
            if (toggle) toggle.remove();
        });
}

if (typeof onAfterUiUpdate === "function") {
    onAfterUiUpdate(eldFitClamps);
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
