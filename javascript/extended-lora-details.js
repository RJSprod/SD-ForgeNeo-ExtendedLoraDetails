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
