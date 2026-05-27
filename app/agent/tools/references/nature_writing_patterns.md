# Nature-Style Scientific Writing Patterns

Extracted from nature-writing v0.2.0 by Yuan1z0825.

## Core Stance

- Author evidence comes first. Do not invent results, mechanisms, references, or limitations.
- Write the argument before writing the sentences.
- Use ambitious but bounded claims.
- If essential evidence is missing, write a placeholder instead of filling the gap.

## Writing Workflow

1. Build one-sentence argument: `In [system/problem], we show [advance] using [approach], supported by [evidence], with [boundary].`
2. Choose section architecture.
3. Map each paragraph to one job: context, gap, approach, result, comparison, mechanism, implication, or limitation.
4. Draft from evidence outward. Keep claims near the data that support them.
5. Calibrate verbs: show, demonstrate, suggest, indicate, enable, may, could.
6. Remove unsupported novelty and universal claims.
7. Paragraph-flow check: one paragraph, one message, clear first sentence, explicit sentence-to-sentence relation.

## Section Defaults

### Abstract (Nature default)
`context/problem -> gap -> approach -> key result -> implication -> boundary`

### Introduction
`field scale -> bottleneck -> prior attempts -> unresolved gap -> present study`
Final paragraph should state what this paper does and how it addresses the gap.
Do NOT summarize all results.

### Results Narrative (evidence ladder)
`system/workflow -> validation -> main result -> baseline comparison -> mechanism/diagnostic -> application/generalization`
Each subsection: claim-first opening, then data support.

### Related Work
`topic scope -> representative methods -> limitation tied to this paper -> distinction`
Group by technical topic and mechanism, NOT by publication year.

### Discussion
`central advance -> evidence meaning -> relation to prior work -> constraints -> future use`
Interpretation and limitations belong here. Do NOT repeat Results.

### Conclusion
`contribution -> decisive evidence -> implication -> boundary`
No new data. No unsupported promises.

### Title
`system/object + action/capability + application or consequence`
Avoid slogan titles and overbroad field claims.

## Literature Review Specific

For review papers (literature reviews):
- Each section covers one theme with synthesis, not paper-by-paper listing.
- Include: convergent findings, divergent findings/debates, methodological observations.
- Gap identification should be specific and actionable.
- End with proposed research agenda.

## Output Format

1. Draft with requested prose.
2. Section outline: 3-7 compact bullets.
3. Assumptions or missing inputs.
4. Claim-evidence map: `Claim: ... | Evidence: ... | Status: supported/needs evidence`.
5. Why this structure: 2-4 short bullets.
