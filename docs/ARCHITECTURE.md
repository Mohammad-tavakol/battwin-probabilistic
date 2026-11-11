# Framework methods and scientific conventions

`Project` provides data, model and workflow services. CLI and notebook calculations use the same APIs. Importing the package does not fit models or execute searches.

## Distributions and calibration

Each target is an equally weighted mixture of 20 XGBoostLSS bootstrap members. Members include their preprocessing, distribution parameters, target transformation and seed. Features and target-specific recipes are stored with the artifact.

For a fitted mixture CDF `F`, validation PIT values determine a monotone map `R`. The calibrated CDF is `R(F(y))`; quantiles use `F_inverse(R_inverse(p))`. Calibration is applied once to the ensemble CDF. The generalised inverse uses the left edge of a plateau. Test responses do not determine the map. Calibration can change medians, means and downstream point estimates; nominal coverage must be evaluated on the specified partition.

## Chain and conditional inputs

The process order is mixing, coating/drying, calendaring and electrochemical. A fixed population of paired particles carries named state variables through the chain. Generated intermediates overwrite their measured counterparts. Member assignments are balanced and retained across stages, with target-specific residual draws. This defines ensemble-induced dependence without fitting a separate residual copula.

Coating/drying porosity conditions on predicted mass loading and thickness. Measured target columns supplied with an input frame are not predictors. Grouped out-of-fold parent predictions define the training recipe; inference constructs parent predictions automatically. Generic training APIs fit direct ensembles on their supplied feature matrices and do not construct this conditional training recipe automatically.

Learned porosities are the default. The optional physical route calculates porosity from paired mass loading, thickness and solid-density values. The physical policy belongs to each chain instance.

## Formulation

The Python model contract uses total-slurry component wt%. `prepare_formulation` derives solid content from active material, conductive additive, binder 1 and binder 2/thickener. Dispersant is separate. Complete compositions are checked for finite, nonnegative values, valid totals and matching material identities. Native partial process tables retain their explicit solid-content input.

The optional [slurry-density helper](SLURRY_DENSITY.md) uses additive constituent volumes, including solvent and every non-solvent component. Its non-solvent total includes dispersant. Slurry density and solid density are distinct inputs.

## Training and evaluation

Group definitions are explicit in each process schema. Repetitions are retained and identical conditions stay within a partition. Tuning uses training data; calibration uses separate validation data. Fixed-configuration cross-validation reports held-out-fold median-based point scores, with an internal calibration holdout in each fold's fitting portion.

## Design, importance and scrap

Optimisation explores development-supported operating conditions with explicit physical, support, tail and optional capacity-cap checks. Inverse design confirms target agreement in two independent particle evaluations. Finite search budgets do not establish a global optimum or prove infeasibility.

Supervised PFI measures the change in point accuracy after permutation. Simulation sensitivity is a separate quantity that does not require measured responses. Scrap calculations distinguish first-stage rejection from flagging with final disposition; parameter flags can overlap. Experimental-reference out-of-tolerance evaluation has a separate reference definition and is not a measured manufacturing scrap rate.

## Numerical reporting

Point estimates distinguish medians from finite-particle means. Interval metrics retain response units. The uncertainty decomposition uses the same sampled particles for aleatoric, epistemic and total variance. The StudentT/log1p target has no finite theoretical moments; sampled means and variances are finite-budget summaries. Record model/data checksums, calibration mode, physical policy, particle count, batch size and seed.
