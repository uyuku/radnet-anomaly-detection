# Repository Hygiene and De-Identification Audit

**Revision Step**: Phase 8, Part C.2  
**Date**: October 31, 2026  
**Scope**: Systematic audit of public files to remove course-format boilerplate, student identification numbers, and personal data.

---

## 1. Audit Summary

To prepare the repository for public release (GitHub and Zenodo) and peer review, all files were audited for institutional course boilerplate, student numbers, and unverified personal identifiers:

| File Audited | Content Identified | Action Taken | Rationale |
|---|---|---|---|
| `README.md` | Course code ("MUH401") and student identification numbers | Removed / replaced with research project metadata | Public documentation should describe research methods and software usage, not course requirements. |
| `zenodo_archive/README.md` | Student ID `2230867059` and course headers | Cleaned; author metadata formatted in standard scholarly CRediT format with ORCID placeholders | Academic deposit standards. |
| `CITATION.cff` | Authors and project metadata | Formatted with scholarly names (Barut & Ergün), affiliations, and version tag `v1.0.0` | Standard citation format for open research software. |
| `docs/*.md` | Historical review notes and temporary agent logs | Audited; historical bug provenance centralized in `CHANGELOG.md` | Preservation of code provenance without cluttering source headers. |
| `report/interim_report.tex` | Course header `MUH401- INTERIM PROJECT REPORT FORMAT`, student ID `2230867059`, framed boxes | Slated for complete removal in Phase 9 (conversion to official MDPI journal LaTeX template). | Journal submission standards. |

---

## 2. Retained Metadata
- **Author Names & Affiliations**: Retained in `CITATION.cff`, `LICENSE`, and author statements (Necdet Ömer Barut, Prof. Dr. Şule Ergün; Hacettepe University).
- **Date & Version**: Version `1.0.0`, released October 31, 2026.
