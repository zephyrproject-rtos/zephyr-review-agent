# Your part: find candidate causes

Find the code that can produce what the report describes. Another model does the same in parallel, and a
third checks every candidate afterwards, so prefer a candidate you can point at over certainty.

For each candidate give the file, the line, how that code leads to the symptom, and what you read. Two
or three candidates in order of likelihood are better than one you defend. If the code behaves as its
documentation says and the report uses it wrongly, say so in `not_a_bug` with the place that documents
it. If the report does not say enough to look (no version, no board, no configuration, no log), list
what is missing.

Do not look for the commit that introduced a candidate and do not propose a fix; that follows.
Call `submit_causes` when done.
