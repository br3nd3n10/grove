# Original planning documents

The initial repository contained only the research plan and short project README. Their exact contents from Git commit `86c3f53` are preserved here before the implementation and experiment documentation is committed.

| Original file | Preserved copy | SHA-256 |
|---|---|---|
| `README.md` | [`docs/original/README.md`](original/README.md) | `b8edd07b5f95f8ee5db37995dcfab52dd53690075ffb57bd718d019011a4d076` |
| `PLAN.md` | [`docs/original/PLAN.md`](original/PLAN.md) | `815d129abd4147c51ef4ebaf1905b4c10a3472e31c144e7636a7213941478769` |

The top-level `README.md` and `PLAN.md` now describe the implemented system and link to the experiment evidence. They do not replace these archived originals.

Git also retains the same originals permanently in commit `86c3f53`:

```bash
git show 86c3f53:README.md
git show 86c3f53:PLAN.md
```

To verify the archive against that commit:

```bash
git show 86c3f53:README.md | sha256sum
git show 86c3f53:PLAN.md | sha256sum
sha256sum docs/original/README.md docs/original/PLAN.md
```
