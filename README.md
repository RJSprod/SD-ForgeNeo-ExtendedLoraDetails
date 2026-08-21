# Extended LoRA Details

A [Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
extension that shows **CSV-sourced Civitai details for a LoRA inside the LoRA
card's own details dialog** — matched to the file by its SHA256 hash.

It extends the existing dialog rather than replacing it: the stock fields
(description, preset, training tags, activation text, preferred weight, notes)
are untouched, and a collapsible **Extended details** panel is added below them —
model description, trigger words and gallery prompts, straight from Civitai.

<!-- ------------------------------------------------------------------ -->

## What it does

* **Hash → CSV lookup.** Every LoRA is identified by the SHA256 of the whole
  file. That digest is looked up in a library of user-supplied CSVs.
* **Extends the details view.** The matched rows are rendered in a collapsible,
  tabbed panel inside the *edit metadata* dialog, so large libraries stay
  navigable: **Overview**, **Description**, **Trigger words**, **Prompts** and
  **Raw CSV row**.
* **The model's own description, in the panel.** Civitai's "about this model"
  text — and the version's own notes — are fetched with everything else and
  shown on their own tab. The HTML is converted to text *once, at fetch time*,
  so the library stays plain CSV and nothing markup-shaped reaches the dialog.
* **Upload CSVs from the settings menu.** *Settings → Extended LoRA Details*
  has a file picker that copies a CSV into the extension's library.
* **On-demand Civitai fetch, in the background.** The *Extended LoRA Details*
  tab runs a fetch over any folder you pick. It hashes each file, resolves it
  through Civitai's `by-hash` endpoint, and writes a CSV into the library —
  on a worker thread, so generation keeps working.
* **Refresh for new content only.** A fetch defaults to *incremental*: any LoRA
  the library has already recorded is skipped — including ones Civitai had
  nothing for — so re-running it on a folder only costs requests for LoRAs you
  actually added. A separate opt-in retries the unresolved ones when you want
  them looked at again.
* **Fills in what is missing.** A refresh also picks up **text details a row
  does not have yet** — a description, say, for rows written before descriptions
  were collected. So pointing this at a folder you have already scanned tops it
  up rather than starting over. Text is fetched by default; **gallery images are
  never downloaded by a fetch** and stay a per-prompt (or per-LoRA) request.
* **Download the gallery image behind a prompt.** Each prompt keeps the image it
  came from; a button beside the prompt saves it into one flat folder. Delete a
  file and it is simply offered again.
* **Standard Gradio components throughout**, styled only with Gradio's own theme
  variables, so your theme keeps applying.

<!-- ------------------------------------------------------------------ -->

## Install

*Extensions → Install from URL*:

```
https://github.com/RJSprod/SD-ForgeNeo-ExtendedLoraDetails
```

or clone into `extensions/`:

```bash
git clone https://github.com/RJSprod/SD-ForgeNeo-ExtendedLoraDetails extensions/SD-ForgeNeo-ExtendedLoraDetails
```

Then restart the WebUI (a *Reload UI* is enough after the first start).

<!-- ------------------------------------------------------------------ -->

## Using it

### 1. Point it at your LoRA folder

Open the *Extended LoRA Details* tab, go to *Fetch from Civitai*, pick the folder
(your Krea 2 folder, say) and press **Start fetch**. Progress and a live log are
on the *Jobs* tab. Everything it finds is written into the extension's own
library directory as a CSV — nothing is written next to your models.

By default it collects the **text** for every LoRA it can resolve: model name,
base model, Civitai link, trigger words, gallery prompts, and the model
description. **Gallery images are not downloaded** — the fetch records each
prompt's image URL, and you ask for the picture itself when you want it.

If you already have a CSV, upload it instead: *Settings → Extended LoRA Details →
Add a CSV to the library*. The library is just a directory of CSVs
(`<extension>/library` by default, changeable in settings); every `.csv` under it
is indexed, recursively.

### 2. Run it again later to fill in the gaps

Re-run the fetch on the same folder. Two things happen, both on by default:

* LoRAs the library already has **complete** details for are skipped, so new
  LoRAs cost the only requests;
* rows that resolved but are **missing text** — no description, typically
  because they were fetched before descriptions were collected — are looked up
  again and topped up. Untick *Fill in text details missing from rows already in
  the library* if you would rather they were left alone.

LoRAs Civitai had nothing for stay skipped either way, so a refresh never
re-asks about every known miss. Tick *Also retry LoRAs Civitai could not resolve
last time* when you do want those tried again.

### 3. Open a LoRA card's details

On the LoRA tab, click a card's ⚙ (*edit metadata*) icon. The **Extended
details** panel sits below the stock fields:

| Tab | Contents |
| --- | --- |
| **Overview** | Every field from the matched row(s): model name, Civitai link, base model, trigger words, and any extra columns your CSV carries. |
| **Description** | The model's Civitai description, as text with its paragraphs, lists and links intact, plus the version's own notes under *About this version*. |
| **Trigger words** | Clickable chips — clicking one adds or removes it from the dialog's *Activation text*. Buttons set or append the whole set. |
| **Prompts** | A searchable selector over every gallery prompt, with ‹ › stepping, a copy button, and *Append to prompt* / *Replace prompt*. Below that, the prompt's gallery image: download it, or the whole set for this LoRA, and see it inline once saved. An *All prompts* accordion lists them. |
| **Raw CSV row** | The matched row(s) as JSON, for when a column is not displayed elsewhere. |

The panel header summarises the match, and the panel opens itself when there is
one (configurable).

<!-- ------------------------------------------------------------------ -->

## CSV format

The reader is deliberately forgiving — column names are matched
case-insensitively and ignoring separators.

| Purpose | Recognised columns |
| --- | --- |
| **Identity (hash)** | `sha256`, `hash`, `file_hash`, `checksum`, `addnet_hash`, `shorthash`, `model_hash`, `sshs_model_hash`, `autov2` |
| **Identity (path)** | `safetensor_file`, `filename`, `path`, `relative_path`, `lora_file`, … |
| Model name | `civitai_model_name`, `model_name`, `name`, `title` |
| Trigger words | `trigger_words`, `trained_words`, `activation_text`, `keywords`, `tags` |
| Description | `model_description`, `description`, `about_this_model` |
| Version description | `version_description`, `about_this_version`, `version_notes` |
| Prompts | `positive_prompt_1..N`, `prompt_1..N`, `prompt` |
| Prompt images | `prompt_image_url_N` / `image_url_N`, `prompt_image_id_N` / `image_id_N` |
| Negative prompt | `negative_prompt`, `negative_text` |
| Link | `civitai_url`, `url`, `model_url` |

**Anything not recognised is still shown**, under the row's field table — no
column from your CSV is dropped.

Matching is tried in this order: **full SHA256 → short/AddNet hash → file path →
file name**. A CSV without a hash column therefore still works; it just matches
by name. To match by hash, include a `sha256` column — the bundled fetcher
always writes one.

### Columns the bundled fetcher writes

```
safetensor_file, sha256, addnet_hash, civitai_model_name, civitai_model_id,
civitai_version_id, civitai_version_name, base_model, civitai_url,
trigger_words, status, note, model_description, version_description,
positive_prompt_1, prompt_image_url_1, prompt_image_id_1,
positive_prompt_2, prompt_image_url_2, prompt_image_id_2, …
```

Each prompt sits next to the gallery image it came from. The three columns are
paired by their trailing number, so an empty prompt cell can never shift an
image onto the wrong prompt.

`model_description` and `version_description` hold **text, not HTML**: Civitai's
markup is converted at fetch time — headings and paragraphs become blank-line
separated blocks, list items get a `•` or a number, and a link becomes
`its text (its url)`. Images inside a description are dropped, since the panel
has nowhere to show them. A description longer than 20 000 characters is cut and
ends in an ellipsis.

A re-fetch replaces the columns above, but **columns you added yourself are kept**
— so annotating a row by hand survives topping it up later.

<!-- ------------------------------------------------------------------ -->

## Command line

The same fetcher runs standalone, without the WebUI:

```bash
python tools/civitai_lora_csv.py --root /path/to/loras
python tools/civitai_lora_csv.py --root /path/to/loras --base-model "Krea 2" --strict
python tools/civitai_lora_csv.py --root /path/to/loras --no-descriptions
```

With no `--output` it writes into the extension's library directory, so the
result is picked up on the next reload. `CIVITAI_API_KEY` is honoured, as is
`--api-key`. Descriptions are collected by default; `--no-descriptions` skips
them, saving one request per model page.

### Identity rules

These are unchanged from the original script this was built from:

* every local file is SHA256-hashed and resolved through
  `/api/v1/model-versions/by-hash/{sha256}`;
* if the matched version exposes SHA256 values, **one of them must equal the
  local digest** — a disagreement is rejected rather than trusted;
* the returned **version id** is authoritative for trigger words and for
  filtering the gallery, so sibling versions on the same model page cannot bleed
  into each other;
* `--base-model` rejects versions Civitai explicitly labels as something else;
  with `--require-base-model-label`, unlabeled exact matches are rejected too.

Gallery pagination is followed to the end, `nextPage` URLs outside `civitai.com`
are refused rather than followed with credentials, and repeated pages stop the
walk defensively. The CSV is written atomically.

<!-- ------------------------------------------------------------------ -->

## Settings

*Settings → Extended LoRA Details*

| Setting | Default | Notes |
| --- | --- | --- |
| Show the extended details panel | on | needs a *Reload UI* |
| Library directory | `<extension>/library` | takes effect immediately |
| Add a CSV to the library | — | the uploader |
| Match by SHA256 first | on | falls back to path, then name |
| Compute a missing SHA256 when the dialog opens | on | turn off and use *Precompute hashes* instead |
| Open the panel automatically | off | |
| Open it automatically when there is a match | on | |
| Prompts rendered in the "All prompts" list | 50 | the selector always reaches every prompt |
| Characters of each prompt in the selector | 90 | |
| Folder for downloaded images | `<extension>/images` | one flat folder |
| Largest image to download | 32 MB | |
| Image download timeout | 30 s | |
| Civitai API key | — | falls back to `CIVITAI_API_KEY` |
| Fetch the model description | on | one extra request per model page |
| Fill in missing text details on a refresh | on | tops up rows that have no description yet |
| Delay between requests / timeout / images per version | 0.15 s / 30 s / all | |
| Only accept this base model | — | e.g. `Krea 2` |
| Require an explicit base-model label | off | |
| Retry unresolved LoRAs on a refresh | off | seeds the checkbox on the fetch tab |

<!-- ------------------------------------------------------------------ -->

## Notes

* **Descriptions cost one request per model page.** The `by-hash` response
  carries the version, not the model page's description, so fetching one means
  asking for the model too. Results are cached per model id inside a run, so
  several versions of the same model still cost a single request. Turn the
  option off if you want the leanest possible scan.
* **A CSV from another tool still works.** It simply has no description column;
  the *Description* tab then says so, and a refresh with *fill in missing text
  details* on adds one.
* **Gallery images need a CSV that has their URLs.** A CSV written before this
  feature existed, or by another tool, has prompts but no image URLs — those
  prompts simply show "no image URL in the library". Re-run a fetch (with
  *retry unresolved* on, or the incremental box off) to pick them up.
* **Whether an image is downloaded is answered from disk**, never from a
  manifest, so moving or deleting files out from under the extension is safe:
  anything missing is offered for download again as if it were new.
* **First open of a large LoRA takes a moment** while its SHA256 is computed.
  The digest is cached in `.cache/hashes.json` (keyed by size and mtime), so it
  only happens once. Run **Precompute hashes** on the extension tab to get it
  over with for the whole collection in the background.
* The SHA256 used here is the digest of the *whole file*. That is deliberately
  not the same as Forge's own LoRA hash, which prefers the kohya
  `sshs_model_hash` from the safetensors header — Civitai's `by-hash` endpoint
  wants the full-file digest. Both are indexed, so CSVs from other tools that
  carry an AddNet hash still match.
* Scans run one at a time on a single worker thread, and are cancellable from
  the *Fetch from Civitai* tab.
* If the built-in LoRA extension is disabled, the panel quietly does not appear;
  the rest of the extension still loads.

<!-- ------------------------------------------------------------------ -->

## Tests

```bash
python tests/test_extended_lora_details.py     # or: pytest tests/
```

These cover CSV parsing and column aliasing, hash/path/name matching, the
Civitai identity rules, incremental refresh and the text backfill, the
HTML → text description conversion and its rendering, atomic CSV writes, and the
job manager. They need no WebUI and make no network calls — the Civitai client is
stubbed.
