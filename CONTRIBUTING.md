# Contributing to HandSpeak 🤟

Thank you for your interest in contributing to **HandSpeak**! This document covers everything you need to get up and running as a contributor.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Submitting Code](#submitting-code)
- [Adding New Signs](#adding-new-signs)
- [Adding a New Sign Language Culture](#adding-a-new-sign-language-culture)
- [Frontend Guidelines](#frontend-guidelines)
- [Commit Message Format](#commit-message-format)
- [Code Style](#code-style)

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/HandSpeak.git
   cd HandSpeak
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feat/my-new-feature
   ```
4. Make your changes, then **push** and open a **Pull Request** against `main`.

---

## Project Structure

```
HandSpeak/
├── backend/
│   ├── main.py              # FastAPI server — add/modify API endpoints here
│   ├── models.py            # PyTorch GCN & MLP model definitions
│   ├── train.py             # Training pipeline (GCN, MLP, Random Forest)
│   ├── generate_data.py     # Synthetic gesture dataset generator
│   ├── rule_classifier.py   # Rule-based finger-extension classifier
│   └── process_kaggle_dataset.py  # Photo dataset → landmark extractor
│
├── frontend/
│   ├── index.html           # Single-page app structure (no inline styles!)
│   ├── styles.css           # Design system — all styling lives here
│   └── app.js               # Frontend logic, MediaPipe, charts, sentence builder
│
├── data/
│   ├── sign_dataset.json    # Augmented synthetic training dataset
│   └── training_history.json# Per-run validation metrics log
│
├── requirements.txt
├── start.py                 # One-command launcher
├── README.md
└── CONTRIBUTING.md          # ← you are here
```

---

## Development Setup

### Prerequisites
- Python 3.10+
- A webcam
- Chrome (recommended) or Firefox

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the development server

```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

The `--reload` flag means the server restarts automatically when you change Python files. Frontend files (`index.html`, `styles.css`, `app.js`) are served statically and just need a browser refresh.

---

## How to Contribute

### Reporting Bugs

Open a GitHub Issue and include:
- Steps to reproduce the problem
- Expected vs actual behaviour
- Browser and OS version
- Console errors (open DevTools → Console)

### Suggesting Features

Open a GitHub Issue with the `enhancement` label. Describe:
- The problem you're solving
- Your proposed solution
- Any alternatives you considered

### Submitting Code

1. Make sure the server starts cleanly with `python3 start.py`
2. Check there are no JS console errors in the browser
3. Open a Pull Request with a clear title and description
4. Reference any related Issues with `Closes #123`

---

## Adding New Signs

1. Open `backend/generate_data.py`
2. Add a template function for the new gesture:
   ```python
   def get_template_mygesture():
       # Return a list of 21 [x, y] landmark coordinates
       # representing the hand pose
       base = get_template_palm()   # start from an open palm
       # modify finger joints as needed …
       return base
   ```
3. Register the sign in `CULTURE_SIGNS`:
   ```python
   CULTURE_SIGNS = {
       "UNIVERSAL": {
           "MyGesture": get_template_mygesture,
           # …
       },
   }
   ```
4. Regenerate the dataset:
   ```bash
   python3 backend/generate_data.py
   ```
5. Retrain the models:
   ```bash
   python3 backend/train.py
   ```
6. Add the sign card to `frontend/index.html` under the relevant culture block:
   ```html
   <div class="sign-card">
     <span class="sign-emoji">✋</span>
     <strong>MyGesture</strong>
     <span class="sign-hint">Brief description of the hand pose</span>
   </div>
   ```

---

## Adding a New Sign Language Culture

1. Add a new key to `CULTURE_SIGNS` in `generate_data.py` (e.g. `"JSL"` for Japanese Sign Language)
2. Add signs following the pattern above
3. Add an `<option>` to both culture dropdowns in `index.html`:
   ```html
   <option value="JSL">JSL (Japanese)</option>
   ```
4. Update `rule_classifier.py` to label matches for the new culture
5. Add a signs block in the Signs Guide tab:
   ```html
   <div class="signs-culture-block" data-culture="JSL">
     <h4 class="signs-culture-title">🇯🇵 JSL — Japanese Sign Language</h4>
     <div class="signs-grid">
       <!-- sign cards here -->
     </div>
   </div>
   ```
6. Add a filter button in `index.html`:
   ```html
   <button class="btn btn-secondary btn-sm" data-filter="JSL">JSL</button>
   ```

---

## Frontend Guidelines

- **No inline styles** — all visual styling belongs in `styles.css`. Use existing CSS classes or add new ones with descriptive names.
- **Use CSS variables** — colours, spacing, and radii are defined as tokens in `:root`. Use `var(--accent)`, `var(--r-md)`, etc. instead of hard-coded values.
- **Keep JS IDs stable** — `app.js` looks up elements by ID. Never rename an existing `id=""` attribute without updating the JS reference.
- **Use semantic HTML** — prefer `<section>`, `<aside>`, `<nav>`, `<header>` over generic `<div>` where it makes sense.

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]
```

| Type | When to use |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code restructure, no feature/fix |
| `perf` | Performance improvement |
| `chore` | Build tools, dependencies |

**Examples:**
```
feat(backend): add Auslan (Australian Sign Language) support
fix(ui): prevent sentence builder from duplicating words on rapid hold
docs(readme): add JSL signs to supported signs table
refactor(css): move theme dot sizes from inline style to CSS class
```

---

## Code Style

### Python
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints where practical
- Keep functions focused and under ~40 lines

### JavaScript
- Use `const` / `let`, never `var`
- Prefer `async/await` over `.then()` chains
- Add a comment above any non-obvious logic block

### CSS
- Group related properties: positioning → box model → typography → visual
- Use `var(--token)` for all colours, radii, and spacing values
- Add a section comment (`/* ── Section Name ── */`) when starting a new component

---

<div align="center">

Made with 🤟 by **[NitheshK4](https://github.com/NitheshK4)** and contributors

</div>
