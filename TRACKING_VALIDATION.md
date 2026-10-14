# Cell tracking validation (2026-10-14)

## Reference and scope

CAFIN `cafin_track.py` was compared with the Notre Dame wound-closure project's
`pipeline/track.py` at reference commit `2318f43`. Before the changes described
here, the feature extraction, overlap/size/centroid score, Hungarian and greedy
matching, bidirectional ID propagation, and gap-closing rules were algorithmically
the same. The CAFIN version mainly changed output packaging. The reference
notebook is Linlin Li's `Cell_Tracking_2D` Colab export; the wound-closure project
uses its matching criteria and adds Hungarian assignment and gap closing.

CAFIN adds an important integration difference: its **Cell tracking** mode
segments raw frames independently, with no tissue registration. Before this
validation, its GUI could select every Nth frame while the matcher still used
pixel overlap, IoU, and centroid proximity. A test across raw frames 0 and 27
linked 198 of 229 starting labels despite substantial tissue deformation;
these identities could not be verified. Cell tracking now requires consecutive
source frames. `Max frames` only truncates the recording and does not skip
frames within it.

## Checks performed

| Input | Check | Observation |
| --- | --- | --- |
| Synthetic labelled cells | Two consecutive frames with overlap | The same numeric ID is retained. |
| Synthetic labelled cells | Large shift or severe size change | The overlap gate rejects the match; proximity alone is not accepted as verified identity. |
| Synthetic labelled cells | Two short fragments and competing gap candidates | A fragment needs at least two observations on each side; only a mutual-nearest eligible pair can be linked. |
| Synthetic labelled cells | Source frame indices `[0, 1, 10, 11]` | A long raw-frame gap is not mistaken for a one-frame dropout. |
| Bundled `demo_data/` | 12 synthetic paired membrane/calcium frames | A mask sequence reconstructed from the demo's known 25-cell motion retained all 25 identities across all 12 frames with no gap links. Cellpose on demo frames 0–1 also produced 25 shared IDs. This is a software check, not biological evidence. |
| Local real trial `trial_5_before` | Two consecutive 256 × 256 central crops, CPU Cellpose 4.2.1 | 324 and 301 segmented regions; the previous gap closer proposed 29 proximity-only links from this two-frame clip. These cannot establish individual identity. |
| Local real trial `lata_trial2_afterdrug` | Frames 27–28 from 256 × 256 central crops, CPU Cellpose 4.2.1 | 215 and 222 segmented regions; 209 starting labels shared an ID, 13 new IDs, and 0 gap links after the fix. These are algorithmic links, not manually verified identities. |
| Same high-motion trial | Diagnostic comparison across frames 0–27, before sparse runs were disallowed | 198 of 229 starting labels shared an ID even though tissue geometry changed strongly. Median matched centroid displacement was only 6.65 px; a separate raw-image feature audit estimated 33.24 px median displacement at frame 27. This demonstrates plausible false continuity from matching by position. The tracker now rejects this sparse input. |

The two real trials are local verification data and are not distributed in this
repository. The crop and the available Cellpose 4.2.1 installation do not match
the full-field, pinned Cellpose 3 workflow, so counts above are diagnostic, not
biological measurements or segmentation benchmarks. No longitudinal manual
cell-ID annotations were available for these trials.

## Safeguards added

- Gap closing now measures missing **source** frames, not missing positions in
  a subsampled stack.
- A gap link requires at least two observations in both fragments and a
  mutual-nearest candidate; two-frame clips cannot create an identity from
  proximity alone.
- CAFIN's cell-tracking GUI and tracker require consecutive source frames.
  Other analysis modes can still subsample for speed.
- Provisional gap links are counted and listed by source frame and distance.
- The GUI warns about frame skipping and very short clips, and its numeric ID
  inspector highlights one cell and shows its centroid path on raw membrane
  frames. The overview palette repeats colors, so color alone is not an ID.
- The frame-0 segmentation/ROI display uses the frame-0 mask even when tracking
  begins from a later reference frame.

Run `python -m pytest tests` to repeat the synthetic and existing unit checks.

## Biological interpretation

The available CAFIN trials are labelled as before/after Latrunculin recordings.
Their images show fin epithelium and a central dark region, but no annotated
wound boundary or verified wound-entry cell identities were supplied. The
Notre Dame wound sequence is a useful **algorithm reference**, not evidence
that a CAFIN track represents one cell entering a wound. That claim remains
unconfirmed.

For a wound-entry result, use a representative wound movie with the raw
membrane channel, run tracking on consecutive frames, inspect each proposed ID
and any gap links on the raw images, and record manually checked cell IDs and
the frames in which they cross a defined wound boundary. Only those reviewed
trajectories should be used for individual-cell migration conclusions.
