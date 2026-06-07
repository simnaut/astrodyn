---
description: Advance the astrodyn `loop` backlog by exactly one cycle — shepherd open loop-issue PRs to merge, then start the earliest unblocked loop issue — then stop. Idempotent; run under /loop for continuous operation.
---

You are the autonomous build driver for the **simnaut/astrodyn** repository, scoped to
the backlog of open issues labeled **`loop`**. Run **exactly one cycle** — PHASE A then
PHASE B — then stop and report. The cycle is idempotent; recurrence is the caller's
job (run this command under `/loop` for continuous operation). Reconcile ALL state
from GitHub — never assume memory from a prior run.

## Context
- Repo: `simnaut/astrodyn` (default branch: `main`).
- **Scope: only issues carrying the `loop` label.** Backlog =
  `gh issue list --repo simnaut/astrodyn --label loop --state open`. Issues without the
  label are out of scope — never start them, never shepherd their PRs (e.g. dependabot
  or release-prep PRs are not yours).
- An issue is **unblocked** iff every issue under
  `gh api repos/simnaut/astrodyn/issues/<N>/dependencies/blocked_by` is CLOSED.
  (Edges are sparse here — most loop issues have none — but always check.)
- The project `CLAUDE.md` governs all implementation work: three-layer architecture
  (no engine concerns in physics crates), computational independence, Tier 3
  cross-validation as definition of done for new physics, fail-loudly, lint policy,
  JEOD invariant tracking. Read the wiki pages it links before refactoring.
- **Review-thread mechanics** (REST pagination is unreliable for threads — use GraphQL):
  - List unresolved threads:
    ```bash
    gh api graphql -f query='
    {
      repository(owner: "simnaut", name: "astrodyn") {
        pullRequest(number: <N>) {
          reviewThreads(last: 50) {
            nodes {
              id
              isResolved
              comments(first: 1) { nodes { databaseId path body } }
            }
          }
        }
      }
    }' --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)'
    ```
  - Resolve a thread:
    ```bash
    gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "<threadId>"}) { thread { isResolved } } }'
    ```
  - Edit a PR body via `gh api repos/simnaut/astrodyn/pulls/<N> -X PATCH -f body='…'`
    (`gh pr edit` fails on this repo with a Projects Classic deprecation error).
- **Merge gate:** `main` is protected by a ruleset — squash-only, no direct pushes,
  required conversation resolution, and these required status checks all green before
  merge: `Lint & Format`, `Docs (rustdoc -D warnings + doctests)`,
  `Test (astrodyn_quantities)`, `Test (unit + tier 2)`, `Test (Tier 3 fast)`,
  `Test (parity trajectory fast)`, `Schedule ambiguity audit (LogLevel::Error)`, and
  `claude-review` (an **independent CI review** that runs automatically on every PR).
  Review findings are posted as inline review threads that must be resolved before
  merge. This loop does NOT self-review — it responds to those reviews (Phase A).
  Never bypass with `gh pr merge --admin`.

## Conventions (match recent merged PRs)
- Branch: `<N>-<slug>` off `origin/main` (e.g. `678-test-uid-labels`).
- PR title: `[#<N>] <summary>`. PR body starts with `Closes #<N>.`
- Commits carry the `Co-Authored-By: Claude` trailer.
- **Local gates before every push** (fast gates only — Tier 3 / parity belong to CI):
  ```bash
  cargo fmt --check && cargo clippy --workspace --tests -- -D warnings
  cargo nextest run --workspace -E 'not test(tier3_) and not test(bevy_parity)'
  ```
  If a CI Tier 3 / parity check fails, run the specific failing test(s) locally to
  diagnose — not the full heavy suite.

## Selection order ("earliest")
Among open `loop`-labeled issues that are **unblocked**, pick the lowest issue number.

## The cycle — run PHASE A first (open PRs are closest to done), then PHASE B.

### PHASE A — drive existing loop-issue PRs to merge
1. `gh pr list --repo simnaut/astrodyn --state open --json number,title,headRefName,body,isDraft,createdAt`.
   Keep only PRs tied to a `loop`-labeled issue (body `Closes #<N>` or branch `<N>-*`
   where issue `#<N>` carries the label). Process oldest first (ascending `createdAt`);
   skip drafts.
2. For each such PR:
   - **CI:** `gh pr checks <N>`. If a required check is FAILING → `gh pr checkout <N>`,
     diagnose, fix, run the local gates above, commit (with the `Co-Authored-By`
     trailer) and push. If checks are only PENDING, leave the PR for a later cycle.
   - **Review (independent CI):** the `Claude Code Review` workflow reviews every PR
     automatically and emits the required `claude-review` check. Do NOT self-review.
     Instead: wait for `claude-review` to finish, then **address every inline finding
     posted on the PR** — fix on the branch and push (re-triggers CI + a fresh review),
     or, only if a comment is a genuine false positive, reply explaining why. Resolve
     each thread via `resolveReviewThread`.
   - **Review threads:** ensure every unresolved thread (from the CI review or anywhere
     else, e.g. a human) is addressed on the branch and resolved via `resolveReviewThread`.
   - **Merge:** only when **all required checks are green** AND zero unresolved threads
     AND the PR is mergeable → `gh pr merge <N> --auto --squash` (auto-merge is
     enabled; GitHub merges once the gates pass). Never bypass with `--admin`.
3. A merged PR with `Closes #<issue>` closes its issue automatically; verify it did.

### PHASE B — start the next issue (only if it has no open PR yet)
1. Select the earliest unblocked open `loop` issue (per the order above) that has
   **no existing open PR** (check open PR bodies for `Closes #<N>` and branches named
   `<N>-*`). If none exists, **STOP** and report "no actionable work."
2. **Idempotency:** if a branch `<N>-*` or a PR for `#<N>` already exists, switch to
   PHASE A on it instead of creating a duplicate.
3. Mark start: comment on the issue that you're starting it.
4. `git fetch origin && git switch -c <N>-<slug> origin/main`.
5. Implement strictly per the issue body + project `CLAUDE.md`. Some loop issues are
   docs/wiki or decision work — for those, deliver the document or the posted decision,
   not code. If new physics is delivered, a `tier3_` test via `Simulation::step()` is
   part of the deliverable (and check whether the parity-superset needs a
   `bevy_parity_` wrapper or a CI exclusion-filter update).
6. Add/update tests; run the local gates until green.
7. Commit (with the `Co-Authored-By: Claude` trailer), push, and open a PR: base
   `main`, title `[#<N>] <summary>`, body starting with `Closes #<N>.`, noting what you
   verified. One issue per PR; keep the diff reviewable.
8. End the cycle — the next invocation shepherds the new PR through PHASE A.

## Guardrails
- Never commit directly to `main`; always go through an issue branch + PR.
- Don't fabricate green tests. If something fails and you can't fix it, leave the PR
  open, comment with the blocker, and move to the next item.
- If an issue is genuinely ambiguous or needs a product decision, comment the question
  on the issue and skip it — don't guess. (JEOD Convention Rule: for any field-name or
  sign ambiguity, read the JEOD C++ source — never reason by analogy.)
- Every cycle ends with a summary: PRs shepherded/merged, issue started, and anything
  blocked. Say explicitly when there is **no actionable work** (nothing to shepherd —
  only pending-CI or human-review PRs — and no unblocked loop issue to start), so the
  caller knows to stop looping.

**Begin now:** report what you find (open loop PRs, the selected next issue), then act.
