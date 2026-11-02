# Project website

This directory contains the REINFORCE Zero project page, including its figures,
math font, styles, and scripts.

- Website: https://hijkzzz.github.io/REINFORCE-Zero/

## Edit and preview

Edit `index.html`, `styles.css`, or `script.js`. After changing CSS or JavaScript,
run this command from the repository root:

```sh
python3 docs/scripts/version_assets.py
```

It generates versioned asset filenames and updates the HTML references. Keep
existing generated files so cached pages can still load their styles and scripts.

Preview locally:

```sh
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000/.

## Publish

In GitHub **Settings → Pages**, choose **Deploy from a branch**, select **main**
and **/docs**, then save. Subsequent pushes to `main` publish the page.
