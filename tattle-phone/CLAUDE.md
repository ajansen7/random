# Tattle Phone

Start by reading `HANDOFF.md` in this folder. It has the full context: decisions,
status, tooling, known traps, and next steps.

Working notes:
- The owner is new to KiCad and PCB work. Explain step by step, and prefer generating
  files/scripts for them over asking them to learn tool internals.
- The routed board in `hardware/kicad/` is now the master copy. **Don't run
  `hardware/kicad_gen/build.py` or `route.py` on it** unless deliberately starting over:
  they regenerate an unrouted board.
- Shared mechanical dimensions live in `hardware/layout.py` (board AND case). After changing it,
  run it, then `make` in `enclosure/`.
- Run `python3 hardware/kicad_gen/check.py` after any board change. It must show
  0 schematic/board mismatches.
- KiCad Python scripting: do any `board.Remove()` calls last (KiCad 7 SWIG breaks after them).
