"""Analysis package for the formal (`formal_osla`) codebase.

Deliberately a top-level sibling of `formal_osla/`, not a subpackage of it:
`formal_osla/` is fingerprinted (sha256 of every `.py`) into every training
manifest, so anything that lives there and changes would invalidate the
provenance of runs already on disk.  Analysis code must be free to evolve.

Ported from `formal_batch/analysis/` (the old 13-input codebase) and adapted to
the new artifact layout.  The old tree is read-only reference; nothing here
writes to it.
"""
