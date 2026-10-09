# Vendored N2M model code

- Source repository: <https://github.com/clvrai/N2M>
- Local source snapshot: `/home/mojie/N2M`
- Source commit: `202bb592cd20d1170fa4af33e5eb14bfd7fba88b`
- Copied concepts/files: `n2m/model/N2Mnet.py` and the PointBERT encoder
  used by `n2m/model/pointbert/{point_encoder,dvae,misc}.py`.

Local changes:

1. imports are contained inside `custom_envs.scripts.N2M.vendor_n2m`;
2. the unused dVAE decoder, plotting helpers and logger/checkpoint-formatting code
   are omitted;
3. `timm.DropPath` is replaced by a small local implementation;
4. checkpoint loading accepts both the upstream `base_model` wrapper and a raw
   state dict and reports incompatible keys without `termcolor`;
5. input validation and CPU-compatible smoke configurations are added.

No `LICENSE` file was present in the inspected upstream snapshot.  Keep this
provenance record with internal copies and confirm redistribution terms before
publishing vendored source.

