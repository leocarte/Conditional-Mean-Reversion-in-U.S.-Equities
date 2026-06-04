# Report

This directory contains the final report source and the curated assets used in
the submitted PDF.

## Main Files

- `report.tex`: LaTeX entrypoint for the final paper.
- `report_body.tex`: main body text included by `report.tex`.
- `report.md`: short prose summary aligned with the final paper.
- `mlfinance_report.sty`: report-specific style file.
- `tables/`: curated CSV tables used in the report.
- `figures/`: curated PNG figures used in the report.
- `report.pdf`: compiled submission PDF.

## Regeneration

If the local walk-forward outputs are available, regenerate the curated report
tables and figures with:

```bash
python scripts/make_final_report_outputs.py
make final-report-outputs
```

Compile the final PDF with:

```bash
make report-pdf
```

## Scope Note

The report uses only course-provided data. The Monthly CRSP extract does not
include `prc`, `shrout`, `shrcd`, or `exchcd`, so full tradability screening is
outside feasible scope and is discussed as a limitation.
