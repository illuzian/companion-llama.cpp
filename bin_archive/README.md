# Archived llama-server builds

Keep the working build before any rebuild. Anthony, 2026-08-07:
*"if you need to compile just make sure you store the current bin separately
with some version note."*

**Archive the whole `bin/` directory, never just `llama-server`.** That file is
a 6.4 KB launcher — the actual server code (and any patch) lives in
`libllama-server-impl.so` beside it. A rollback that copies only the
executable restores nothing.

---

## build-crossslot-bin.b10194-3aab91e16.crossslot.20260807.tar.zst

| | |
|---|---|
| archived | 2026-08-07 08:25 AWST |
| source | llama.cpp `b10194-3aab91e16` |
| patch | `companion_app/scripts/patches/llama-server-cross-slot-prefix-adoption.patch` |
| built | gfx1100, local hipcc, `-DCMAKE_HIP_ARCHITECTURES=gfx1100` |
| size | 16 MB compressed / 91 MB on disk |
| sha256 | `689be103c10a5b1ddee5779bf3de03b8ca66a83ac5ecccede5fcdcf32988add5` |

**Status — this is the build that served her on 2026-08-07**, and it is the
last known-good one. Cross-slot prefix adoption is compiled in but was
measured **never firing in production**: zero `adopting shared prefix` events
across the entire server log.

Cause (measured 2026-08-07, not inferred): the lane keeper's rendered
`[head + lane]` on slot 3 is **larger than a mirror's entire prompt** —
slot 3 peaked at 53,954 tokens against slot 1's whole 47,889-token prompt. The
patch's pure-ancestor guard is

```c
if (lcp + 1 < other.prompt.tokens.size()) continue;
```

so a donor longer than the prompt it is meant to prefix can never qualify. The
patch is correct; the keeper is not rendering the mirrors' prefix. Fix the
keeper before concluding anything about the patch.

### Rollback

```sh
cd ~/.unsloth/llama.cpp
tar --zstd -xf bin_archive/build-crossslot-bin.b10194-3aab91e16.crossslot.20260807.tar.zst \
    -C build-crossslot
```

Verify with the recorded sha256 before extracting.
