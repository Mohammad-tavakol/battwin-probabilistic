# Slurry density and material properties

The additive-volume estimate is `rho = 1 / sum(w_i / rho_i)`. Each `w_i` is a mass fraction of total slurry and `rho_i` is the constituent density in g/cm3. Solvent is included. The calculation assumes additive constituent volumes; it is distinct from a measured slurry density.

| Material | Density (g/cm3) |
| --- | ---: |
| NMC622 | 4.447 |
| NMC111 | 4.77 |
| Graphite | 2.26 |
| C65 | 1.6 |
| C45 | 1.6, assumed |
| PVDF | 1.78 |
| SBR, dry polymer | 1.3 |
| CMC | 1.6 |
| Water | 1.0 |
| NMP | 1.03 |
| Unspecified material | 1.6, explicit fallback assumption |

Use grade- and temperature-specific values when available. Water in a polymer dispersion belongs in the solvent fraction.

`estimate_slurry_density(values, material_densities=None)` accepts five total-slurry component slots, their material identities, solvent identity and solid content. Unused slots are zero. This calculation's solid content includes all non-solvent constituents, including dispersant once. Invalid or incomplete compositions are rejected without normalisation.

```python
from battwin_probabilistic.slurry_density import estimate_slurry_density
values = {
    "Chemistry": "NMC622", "Am/total weight%": 57.6,
    "Conductive additive type": "C65", "Conductive additive/total weight %": 1.2,
    "Binder1 type": "PVDF", "Binder1/total weight %": 1.2,
    "Binder2 (thickener)/total weight %": 0., "Dispersant additive (%)": 0.,
    "Solvent": "NMP", "Solid Content (%)": 60.,
}
result = estimate_slurry_density(values, {"PVDF": 1.78})
print(result["density_g_cm3"])
```

The result records material values, volume contributions and assumptions. Overrides must be finite and positive. Convert dry-solid percentages first using `dry_wt_pct * solid_content_pct / 100`.

`apply_density` is an explicit adapter: set `slurry_density_mode="estimated"`, with material overrides under `slurry_density_materials`. It retains the manual value under `manual_slurry_density_g_cm3`. Normal prediction calls use the supplied density. The formula identifier is `mass_fraction_additive_volume_v1`.

The separate Python `prepare_formulation` helper derives the model solid-content feature from four fields and keeps dispersant separate. Use the input definition required by the selected interface. Solid-density model features are independent of this estimate.
