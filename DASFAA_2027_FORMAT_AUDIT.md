# DASFAA 2027 format audit

The anonymous LNCS conversion is [ALPS_DASFAA_2027/main.pdf](ALPS_DASFAA_2027/main.pdf).
It uses `llncs.cls`, anonymous author metadata, and the Springer bibliography
style. The compiled PDF has 23 pages and therefore exceeds the DASFAA research
track limit of 16 pages including references. It is a source conversion draft,
not a submission-ready PDF.

The conversion incorporates the prospective holdout, graph-weight sensitivity,
the measured retrieval/interactions/load trade-off, and the bounded Ethereum
receipt retry results. The IEEE manuscript remains the compact 10-page version.
Reducing the LNCS version to 16 pages requires substantive selection of figures,
tables and exposition; shrinking fonts or margins would violate the template.

## Checks

- `llncs.cls` and `splncs04.bst` are local and compile successfully.
- PDF metadata author is `Anonymous`.
- No unresolved citation markers remain.
- Four minor overfull boxes remain in the long LNCS conversion and should be
  corrected during the final 16-page editorial cut.
- The working venue is DASFAA 2027; the PDF does not claim page-limit compliance.
