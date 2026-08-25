# Phase 9C router checksum investigation

`config/router_frozen_v1.yaml` has one repository commit: its introduction in
`ff5af0c`. Git history and the Phase 9C starting worktree show no later router
change. The extraction-v2 release manifest was introduced in the same commit
with a SHA-256 value that does not match the committed router bytes.

The manifest hash `0580cca6...a0c5` matches the Git-canonical LF bytes. The
Windows checkout materialized CRLF bytes, producing `8e6080c7...71e4`; every
other governed YAML showed the same pattern and matched after LF
canonicalization. This was therefore a cross-platform verifier defect, not a
stale checksum or a runtime policy mutation. Phase 9C makes governed text
hashing line-ending invariant. No router configuration, threshold, weight, or
behavior changed.
