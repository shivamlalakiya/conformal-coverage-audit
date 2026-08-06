# What was fixed before the data, and what was not

A pre-registration written after the analysis cannot be verified as one, and this
document does not pretend otherwise. It exists because the alternative — leaving the
two categories mixed — lets every choice read as though it had been planned. What
follows separates the decisions taken before any measurement from the decisions taken
after seeing output, names the second kind, and says what each one would have looked
like had it gone the other way.

The honest summary: the arithmetic results were fixed in advance and could not have
been tuned. Several of the *measurement* choices were not, and two of them changed a
reported number.

---

## Fixed before any measurement

These follow from the definition of the object and admit no discretion. Each is
exact-arithmetic and reproducible from the definitions alone.

| Decision | Where it is fixed |
|---|---|
| The required index is the least rank whose exact coverage reaches the request | Derived from the order-statistic identity, not chosen |
| Feasibility is a strict inequality on the calibration size | Same derivation |
| A two-rail interval owes a *span*, and both rails are finite only above twice the one-sided floor | Same derivation, stated as one proposition covering the general division |
| The thirteen definitions to audit | Hyndman and Fan's catalogue plus every alias the library exposes; the set is somebody else's and we did not select within it |
| The rule that a site is *filed* when what ships and what is documented disagree | Written down before the first report and applied unchanged; it is what excludes the largest measured shortfall from the filings |
| Exact rational arithmetic everywhere, no floating-point comparison in a classification | Fixed at the start after a float comparison produced a wrong class early on |
| Every number in the manuscripts comes from a committed probe output, and a hand-typed decimal fails the build | Gate, not intention |

None of these could have been adjusted to improve a result, because none of them is
fitted to anything. That is the point of putting them in a separate list.

---

## Decided after seeing data

Each row names the decision, when it was made, and the counterfactual.

### 1. The calibration sizes used in the coincidence-band comparison

**After.** The paired comparison contrasts sizes where the required and delivered
ranks coincide against sizes where they differ. Which sizes those *are* is exact
arithmetic and was not chosen. Which ones to *run the forecaster on* was chosen after
seeing that the coincidence pattern was periodic — we picked sizes on and off the
residue class deliberately, in order to make the contrast visible.

**Counterfactual.** A grid chosen without that knowledge would have landed
haphazardly across the two regimes and the paired difference would have been noisier
in a way that flatters nothing. This choice sharpens a comparison; it does not
create one.

**And it went wrong once, in the other direction.** A first grid for the SQL probe was
`{10, 20, 50, 100}` — chosen for roundness, before looking. None of those sizes lies
on the delivering residue class, so the probe reported that a convention "never
delivers" when it delivers one time in ten. The grid was chosen by the author and the
correction was forced by the bug. This is the clearest case in the project of
round-number choices hiding a phenomenon, and it is why every sweep now carries sizes
on *and* off each level's residue class.

### 2. The non-unit-fraction level

**After.** Early sweeps used only the three round levels a practitioner asks for. The rule that every sweep must
also carry a level whose complement is not a unit fraction was added after noticing
that two conventions are indistinguishable at every one of those and part company
immediately off them.

**Counterfactual.** Without it, two distinct conventions would have been reported as
indistinguishable, a period would have been reported as a decade when it is not, and
one claim about attainability would have been reported as universal when it is
level-dependent. Three separate results depended on this choice, all of them in the
direction of *less* generality than the first draft claimed.

### 3. Which distributions the tail sweep uses

**After.** Normal, lognormal, exponential and Pareto were fixed early. The exponential
and Pareto were added after the first two, on noticing that a sweep containing no
exactly-generalised-Pareto case could not distinguish "the closed form is right" from
"the closed form is approximately right for everything we tried".

**Counterfactual.** The sweep would have supported the law without being able to test
the hypothesis it assumes. Adding them made a check possible that had not been
available, and the check passes exactly for the two GPD cases and approximately for
the other two, which is the informative outcome.

### 4. The depth grid in the tail sweep

**After.** Extended from a first grid after the first grid's largest depth turned out
to be inside the regime where the second-order term still mattered. The standing rule
that came out of it — push a sweep past double whatever edge it first runs into —
was written down at that point and has been applied since.

**Counterfactual.** The rate would have been reported with a remainder that looked
larger than it is, and the second-order coefficient would not have been derived at all.

### 5. The set of packages audited

**Partly after.** The criterion — whether a package turns a miscoverage target into a position in a
held-out sample — was fixed first. Which packages satisfy it was discovered by reading
them. Three were excluded, with the reason stated per package, and the exclusions were
decided after reading the source because they could not have been decided before.

**Counterfactual.** Nothing here is tunable: the criterion admits or excludes, and a
reader can check either judgement against the named file. The download-share figure is
reported precisely so this list is not taken on trust.

### 6. The mutation set that tests the gates

**Entirely after, and necessarily so.** Each mutation was written to attack a gate
that already existed. Three of the first mutations tested nothing — two were planted
in LaTeX comments that the gate strips before reading, one in a helper no gate calls —
and the harness scored them as caught. They were rewritten to plant in live prose and
in a called helper.

**Counterfactual.** The gate suite would have reported a higher catch rate than it
earned. This is the one item in this document where the *first* version of the
measurement was actively misleading rather than merely less informative.

---

## Decisions taken after seeing data that changed a published number

Two, and both are recorded in the manuscripts at the point where the number appears.

1. **A ratio derived by hand from two rounded fields.** Two macros were computed by
   dividing values already rounded for display, and shipped values the probe had never
   printed. The fix was a gate forbidding arithmetic on a parsed field, which is now
   the only gate whose existence is owed to a specific error rather than to a general
   principle.

2. **A uniformity claim supported by a grid.** A derivative bound was asserted to hold
   uniformly on the strength of a numerical sweep. The sweep was correct and the
   inference was not; the quantity it relied on exceeds the claimed bound by a factor
   of eighteen at one end of the parameter range. It was replaced by a proved bound,
   and the retracted wording is now one of the phrases the build refuses to reinstate.

---

## What this document cannot do

It cannot make the pre-registration verifiable. A reader has the commit history, which
shows when each probe and each gate appeared, and that is the only checkable version of
this claim. Where the history and this document disagree, the history is right.

It also does not cover the writing. Which findings became a section and which became a
sentence was decided throughout, by judgement, and no rule was written down for it.
