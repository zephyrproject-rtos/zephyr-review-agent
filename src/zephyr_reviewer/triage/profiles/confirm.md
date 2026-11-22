# Your part: decide, trace, and propose the triage

Candidate causes from a first pass are listed below, when there are any. They are leads, not findings:
check each against the code yourself, and look further if none holds.

1. **Decide.** One verdict:
   - `confirmed`: you read the path from the reported use to the failure, and the code is wrong.
   - `plausible`: a likely cause, with a link in the path you could not check. Name that link.
   - `not_a_bug`: the code behaves as documented; say where that is documented.
   - `needs_info`: the report lacks what is needed. Ask for exactly that, as questions.
   - `cannot_confirm`: you looked and found no cause. Say what you checked.
2. **Trace.** For a confirmed or plausible cause, find the commit that introduced it, as the working
   rules say. Leave `culprit` out when the history does not show it.
3. **File it.**
   - `labels`: the areas and labels are taken from MAINTAINERS for the file you name as the cause,
     so naming the right file is what files the issue. Add a label only when another area is
     involved as well, for example the board when the defect is specific to one, and take it from
     what the `maintainers` tool lists for that path. A label MAINTAINERS does not define is dropped.
   - `priority`, with the reason in one sentence:
     - `high`: a crash, hang, memory corruption, data loss or security problem in a commonly used
       configuration, or a regression that breaks a supported platform or a release.
     - `medium`: a feature does not work as documented, the system is otherwise usable, and there is
       no easy workaround.
     - `low`: a minor or cosmetic problem, a rare configuration, or one with an easy workaround.
     The impact the reporter chose is their view; weigh it against what the code shows.
   - `duplicates`: issues `search_issues` finds that report the same defect, not merely the same area.
4. **Fix.** Describe the smallest correct change. Give a `patch` only when it is a few lines and you
   have read every caller it affects.

Call `submit_triage` when done.
