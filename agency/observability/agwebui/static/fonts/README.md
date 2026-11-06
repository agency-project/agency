# Agency fonts

Unmodified variable WOFF2 assets from Vercel's official [`geist`](https://www.npmjs.com/package/geist) package, version 1.7.2.
Both families are licensed under the SIL Open Font License 1.1; see [LICENSE.txt](LICENSE.txt).

Package archive: https://registry.npmjs.org/geist/-/geist-1.7.2.tgz
Archive integrity (verified before extraction): `sha512-Gu5lDFa3pLRyoBlBPf0QIFHVdWAnpco7fS1bJm41jyLPFoguBgiubseUN2oLXMgqZ7uxAxDoXcHMhCY/fOTTgg==`

Source paths:
- `dist/fonts/geist-sans/Geist-Variable.woff2`
- `dist/fonts/geist-mono/GeistMono-Variable.woff2`

These files are served locally by Agency's existing `/static` route and included in its Python wheel.
No npm, Next.js, external font service, or font download is required at runtime.
To update, download a pinned release of the official package, verify its registry integrity,
and replace these two files and the package license together. Update the version and integrity above.
Typography roles, weights, and line heights are defined in `../theme.css`.
