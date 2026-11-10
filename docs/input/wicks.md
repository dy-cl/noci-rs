# Wick intermediates: `wicks`

The optional `wicks` table controls the [extended nonorthogonal Wick
intermediates](../methods/elements.md#different-frames-extended-nonorthogonal-wicks-theorem).

```lua
wicks = {
    enabled = true,
    compare = false,

    storage = "disk",
    cachedir = "/scratch/me/wicks",
}
```

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` without a `wicks` table, `false` when omitted from one | Use Wick intermediates for matrix elements between frames. **Set it explicitly** if you give a `wicks` table: see [Parsing rules](index.md#parsing-rules). |
| `compare` | boolean | `false` | Check every Wick matrix element against the generalised Slater–Condon result. Also builds the intermediates. For testing only. |
| `storage` | string | `"ram"` | `"ram"`, shared between MPI ranks on a node, or `"disk"`, a memory-mapped cache. |
| `cachedir` | string | `<write.write_dir>/wicks` | Directory of the disk cache. It also holds the disk-backed QMC factor tables and SNOCI `full_m` and `factor_tables` files. |

Wick intermediates are needed by the stochastic factor tables, the factorised SNOCI path and
B-apply with more than one parent, so leave them enabled unless you are testing.
