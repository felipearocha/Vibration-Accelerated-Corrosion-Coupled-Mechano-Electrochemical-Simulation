# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- engine: one-year depth is the mean rate times one year (was multiplied by 8766 hours)
- engine: overpotential is now a signed anodic overpotential (was |eta| of a cathodic value), so corrosion rate rises as pH falls
- engine: anodic Tafel term is now 10^(eta/ba); the slope was inverted (10.8 /V, now 19.2 /V)
- tests: suite is now collected by pytest (19 tests); it crashed pytest before and CI masked it with `|| true`
- run_pipeline: no longer aborts on the missing LinkedIn file or on a non-UTF-8 console
- models: re-saved gbr_model.pkl and scaler.pkl, which no longer loaded with current scikit-learn (predictions unchanged)

### Added
- run_all.py entry point (alias of run_pipeline.py)

### Docs
- docs: insert real Zenodo DOIs (concept 10.5281/zenodo.20172485, version 10.5281/zenodo.20172486)

### CI
- ci: drop Python 3.10 from matrix (modern scipy/numpy require >=3.11)
- ci: test failures now fail the job; bump checkout and setup-python to v7
- ci: Dependabot keeps pip floors unless needed and groups Actions updates
## [0.1.1] - 2026-05-13

### Changed
- chore(release): prep v0.1.1 — Zenodo metadata, DOI badge, How to Cite
- chore(meta): add CI matrix, Dependabot, and Integrity Code Series cross-links

### Other
- Update README.md

[0.1.1]: https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation/releases/tag/v0.1.1
## [0.1.0] - 2026-03-11

### Added
- Add files via upload

### Other
- Update README.md
- Initial commit

[0.1.0]: https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation/releases/tag/v0.1.0
