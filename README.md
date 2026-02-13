# lol-splash-collector

Builds and updates a local splash metadata database from the LoL wiki, with optional image downloads.

## Prebuilt Dataset (Hugging Face)

You can download a prebuilt copy from [rconhf/lol-splash on Hugging Face](https://huggingface.co/datasets/rconhf/lol-splash) and place it in `./output/dataset`.

This can be used:
- as a base for incremental updates with this tool, or
- on its own if you only need the collected files.

**Status:** up to date as of **13/02/2026**.

## What it does

- Fetches `Module:SkinData/data` and `Module:Filename` revision metadata.
- Parses every champion + skin entry and preserves full skin metadata.
- Resolves splash URLs using candidate order:
  1. `_HD.jpg`
  2. `_HD.png`
  3. `.jpg`
  4. `.png`
- Matches skins by **resolved URL** to detect shared splash artwork groups.
- Picks canonical metadata per resolved image using the **oldest release date** in that URL group.
- Writes update/diff/version information so reruns can detect changes.

## Run metadata build

```bash
python build_splash_metadata.py
```

Optional:

```bash
python build_splash_metadata.py --output-dir output --metadata-name metadata.json
```

## Run dry-run dataset materialization

Creates clickable `.url` shortcuts and per-art metadata files, but no image downloads:

```bash
python build_splash_metadata.py --materialize-assets --dry-run
```

## Run real downloads

Downloads image files to `output/dataset/...` and writes per-art metadata files:

```bash
python build_splash_metadata.py --materialize-assets
```

Useful controls:

```bash
python build_splash_metadata.py \
  --materialize-assets \
  --download-workers 4 \
  --max-requests-per-second 2.0 \
  --download-retries 4 \
  --retry-backoff-seconds 1.0 \
  --download-timeout-seconds 60 \
  --force-redownload
```

Optional strict validation mode (usually not needed with CDN-transcoded media):

```bash
python build_splash_metadata.py --materialize-assets --strict-size-check
```

Optional bounded run for testing:

```bash
python build_splash_metadata.py --materialize-assets --max-downloads 25
```

## Output

- `output/metadata.json`  
  Full database, source module revision info, run stats, and diff from previous local run.
- `output/dataset_dryrun/...` (dry-run mode)  
  Per-art dry-run output:
  - `<image>.url` (browser shortcut)
  - `<image>.metadata.json` (canonical metadata for that image)
- `output/dataset/...` (real mode)  
  Per-art download output:
  - `<image>` (downloaded image)
  - `<image>.metadata.json` (canonical metadata for that image)

## Notes

- Images are downloaded only when `--materialize-assets` is used without `--dry-run`.
- `.url` shortcut files are created only in dry-run mode (`output/dataset_dryrun/...`).
- Grouping uses resolved URL returned by MediaWiki `imageinfo`.
- A URL group is treated as shared only when it maps to more than one unique champion.
- Before every overwrite of `output/metadata.json`, the previous file is automatically backed up to `output/backups/metadata/`.
- Downloading is conservative by default (4 workers, 2 requests/sec global cap, retries with backoff).
- Existing files are skipped by default when already present; use `--force-redownload` to override.
- Retry attempts happen immediately per file (same worker task), not in a separate end-of-run retry queue.
- Failed downloads are reported in end-of-run totals and per-file `<image>.metadata.json` with status/error details.
- The wiki/CDN can serve a valid image with byte size different from API `imageinfo.size`; by default this is accepted and recorded as `downloaded_size_mismatch`.
- You can rerun the script anytime after wiki updates; it rewrites metadata and tracks changes.
