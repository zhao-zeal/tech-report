# Reference

- Retained DOCX: `E:\电量预测赛道\技术报告\output\docx\地市用电量预测初赛技术报告_20260916_代码排版修复版.docx`
- SHA-256: `C522C63E2E787EC972FC0C853FFB687F102DD51550A166A9FB86DD444D61A616`
- Rendered page count: 15
- Section count: 1
- Render evidence: `E:\电量预测赛道\技术报告\qa_pv_report\template-render3`
- Style evidence: `E:\电量预测赛道\技术报告\qa_pv_report\template-style-evidence.json`

# Page system

- Letter portrait, 8.5 × 11 inches.
- One section with 1.0 inch margins on every side.
- No visible header or footer furniture and no page number.
- Continuous body flow, with explicit page breaks before the full-page structure figure and each appendix code subsection.

# Typography

- Main title block centered and black. First line 22 pt, second line 22 pt, task subtitle 16 pt.
- Chapter headings use Chinese numeric labels, black, bold, 16 pt, left aligned.
- Subsection headings use full-width parenthesized Chinese numbers, black, bold, 14 pt, left aligned.
- Body text uses a Chinese serif family compatible with SimSun, 12 pt, black, 1.55 line spacing, justified, first-line indent of two Chinese characters.
- Captions use black bold text, 11 pt, centered, and stay with their figure or table.
- Code uses Consolas, about 9 to 10 pt, single spacing, no first-line indent.
- Equations are centered native Word math objects.

# Lists and tables

- Tables use the full text width only when necessary, with light gray internal and outer borders.
- Header rows use pale blue-gray fill with bold black text.
- Narrative columns are left aligned; numeric and short-value columns are centered.
- Rows expand naturally; no fixed row height. Repeat the header on multi-page tables.
- Table titles appear immediately above the table and stay with it.

# Components and flow

- Opening title block followed immediately by chapter one on page 1.
- Five chapters in the sequence requested by the user.
- Main structure figure is centered on a dedicated or nearly dedicated page and followed by a centered caption.
- Appendix code excerpts begin on clean pages. Each excerpt has a numbered bold lead and a short source-oriented explanation.
- References appear as numbered Chinese bibliography paragraphs in the final appendix subsection.

# Slot map

- Replace the full city-task body with the photovoltaic-task report while preserving the document theme, styles, section geometry, and package-level font/theme definitions.
- Replace the city model figure and all city-specific tables, equations, captions, and code with photovoltaic evidence.
- Preserve no city-task prose, scores, model names, identifiers, or references.
- Reuse the source document's title, heading, body, table, caption, equation, and code page patterns.

# Package preservation

- Preserve the original file unchanged.
- Start from a byte copy of the reference and replace only the main body content plus image relationships required for the new structure figure.
- Preserve styles, theme, numbering definitions, settings, core document relationships, and section geometry unless a report-specific element requires an additional relationship.

# Fidelity gates

- Reference SHA-256 must remain unchanged.
- Final geometry must remain Letter portrait with one-inch margins.
- Final rendering must show black headings, readable 12 pt body text, gray bordered tables, centered native equations, and Consolas code pages.
- Inspect every final page at 100 percent zoom for clipping, overlap, broken tables, blurred figure text, missing equation glyphs, isolated headings, and accidental city-task residue.
