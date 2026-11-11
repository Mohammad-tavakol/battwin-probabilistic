"""Stage-wise first rejection and flag-only final-disposition policies."""
from pathlib import Path
import numpy as np
import pandas as pd
from .design_space import OUTPUTS,GRAV,VOL,DRY_THICK,SearchConfig
def policy_masks(stage_bad, final_bad):
    alive = np.ones(len(final_bad), bool)
    first = np.full(len(final_bad), "", object)
    history = {}
    for stage in OUTPUTS:
        new = alive & stage_bad[stage]
        first[new] = stage
        alive &= ~stage_bad[stage]
        history[stage] = (new, alive.copy())
    direct = alive & ~final_bad
    flagging = ~final_bad
    assert np.all(direct <= flagging)
    assert sum(int(v[0].sum()) for v in history.values()) + int(direct.sum()) == len(final_bad)
    return first, history, direct, flagging

class ScrapPolicyMixin:
    def physical_rule_masks(self, particles, verify_orchestration=True):
        rules = []
        def add(stage, parameter, rule, passed):
            rules.append((stage, parameter, rule, ~np.asarray(passed, bool)))
        def v(c):
            return particles[c].to_numpy(float)
        for stage, outputs in OUTPUTS.items():
            for c in outputs:
                add(stage, c, "finite and nonnegative", np.isfinite(v(c)) & (v(c) >= 0))
        m, t, dp = v("Mass loading dry (mg/cm2)"), v(DRY_THICK), v("Dried Porosity (%)")
        ct, cp = v("Calendered thickness (um)"), v("Calendered porosity (%)")
        add("Drying", DRY_THICK, "Dried thickness < coating gap", t < v("Coating Gap (micron)"))
        add("Drying", DRY_THICK, "Positive dried thickness and density", (t > 0) & (v("Solid density reference (g/cm3)") > 0))
        add("Drying", "Dried Porosity (%)", "Dried porosity in [0,100]", (dp >= 0) & (dp <= 100))
        add("Drying", "Dried Porosity (%)", "Dried porosity > 30%", dp > 30)
        add("Calendaring", "Calendaring Gap (um)", "Roll gap < dried thickness", v("Calendaring Gap (um)") < t)
        add("Calendaring", "Calendered thickness (um)", "Calendered thickness < dried thickness", ct < t)
        add("Calendaring", "Calendered thickness (um)", "Positive calendered thickness", ct > 0)
        add("Calendaring", "Calendered porosity (%)", "Dried porosity > calendered porosity", dp > cp)
        add("Calendaring", "Calendered porosity (%)", "Calendered porosity in [0,100]", (cp >= 0) & (cp <= 100))
        add("Calendaring", "Calendered porosity (%)", "Calendered porosity > 16%", cp > 16)
        add("Electrochemical", GRAV, "Gravimetric capacity < 500 mAh/g", v(GRAV) < 500)
        if verify_orchestration:
            expected, _ = self.context["physical_stage_flags"](particles, self.line.stages)
            for stage in OUTPUTS:
                calculated = np.logical_or.reduce([fail for s, _, _, fail in rules if s == stage])
                np.testing.assert_array_equal(calculated, expected[stage])
        return rules

    def assess_scrap(self, particles=None, tolerance_pairs=None,
                     report_prefix="alternative", verify_orchestration=True,
                     save_physical_rules=True):
        """Score saved particles for explicit (intermediate, final) tolerances.

        Defaults preserve the original intermediate-only sensitivity experiment.
        A matched sweep passes (tau, tau), without changing inverse selection.
        """
        cfg = self.config
        accepted = set(self.alternatives.path_id.astype(int))
        if particles is None:
            particles = pd.concat([pd.read_pickle(f) for f in self.verification_files], ignore_index=True)
            particles = particles.loc[particles.path_id.isin(accepted)].copy()
            particles.to_pickle(self.out / "reports/alternative_particles.pkl.gz", compression="gzip")
        particles = particles.loc[particles.path_id.isin(accepted)].copy()
        tolerance_pairs = ([(tau, cfg.target_tolerance) for tau in cfg.intermediate_tolerances]
                           if tolerance_pairs is None else list(tolerance_pairs))
        if not tolerance_pairs or any(not (0 < a < 1 and 0 < b < 1) for a, b in tolerance_pairs):
            raise ValueError("Tolerance pairs must contain fractions between zero and one.")
        stages, parameters, physical_rows, rankings = [], [], [], []
        for pid, p in particles.groupby("path_id", sort=True):
            p = p.reset_index(drop=True)
            n = len(p)
            means = {c: float(p[c].mean()) for cols in OUTPUTS.values() for c in cols}
            physical_rules = self.physical_rule_masks(p, verify_orchestration=verify_orchestration)
            physical_stage = {s: np.logical_or.reduce([f for stage, _, _, f in physical_rules if stage == s]) for s in OUTPUTS}
            for stage, parameter, rule, fail in physical_rules:
                physical_rows.append(dict(path_id=pid, stage=stage, parameter=parameter, rule=rule,
                                          failures=int(fail.sum()), n_particles=n, failure_pct=100*fail.mean()))
            for tau, final_tau in tolerance_pairs:
                bad = {s: physical_stage[s].copy() for s in OUTPUTS}
                per_parameter = {}
                for stage, cols in OUTPUTS.items():
                    for c in cols:
                        centre = float(self.target[[GRAV, VOL].index(c)]) if stage == "Electrochemical" else means[c]
                        tol = final_tau if stage == "Electrochemical" else tau
                        values = p[c].to_numpy(float)
                        tolerance_fail = (~np.isfinite(values) | (values < centre*(1-tol)) | (values > centre*(1+tol)))
                        phys = np.logical_or.reduce([f for s, par, _, f in physical_rules if s == stage and par == c])
                        per_parameter[(stage, c)] = (tolerance_fail, phys, tolerance_fail | phys, centre, tol)
                        bad[stage] |= tolerance_fail
                    # Include the roll-gap physical input separately from predicted outputs.
                    if stage == "Calendaring":
                        phys = np.logical_or.reduce([f for s, par, _, f in physical_rules if s == stage and par == "Calendaring Gap (um)"])
                        per_parameter[(stage, "Calendaring Gap (um)")] = (np.zeros(n, bool), phys, phys, np.nan, np.nan)
                final_bad = bad["Electrochemical"]
                first, history, direct, flagging = policy_masks(bad, final_bad)
                rank = dict(path_id=pid, intermediate_tolerance_pct=100*tau, final_tolerance_pct=100*final_tau,
                            n_particles=n, grav_mean=means[GRAV], vol_mean=means[VOL],
                            direct_yield_pct=100*direct.mean(), flagging_yield_pct=100*flagging.mean(),
                            direct_passes=int(direct.sum()), flagging_passes=int(flagging.sum()),
                            recoverable_pct_original=100*(flagging & ~direct).mean())
                rankings.append(rank)
                alive_before = np.ones(n, bool)
                for stage in OUTPUTS:
                    newly, alive_after = history[stage]
                    count = int(bad[stage].sum())
                    recovery = float((bad[stage] & flagging).sum()/count) if count else np.nan
                    for policy in ["direct_reject", "flag_only"]:
                        stages.append(dict(path_id=pid, policy=policy, stage=stage,
                                           intermediate_tolerance_pct=100*tau, final_tolerance_pct=100*final_tau,
                                           n_particles=n, stage_flag_pct=100*bad[stage].mean(),
                                           stage_scrap_pct_original=100*(newly.mean() if policy == "direct_reject" else final_bad.mean() if stage == "Electrochemical" else 0),
                                           survival_pct_original=100*(alive_after.mean() if policy == "direct_reject" else flagging.mean() if stage == "Electrochemical" else 1),
                                           stage_rejection_pct_arrivals=100*newly.sum()/alive_before.sum() if policy == "direct_reject" and alive_before.sum() else 100*final_bad.mean() if policy == "flag_only" and stage == "Electrochemical" else 0 if policy == "flag_only" else np.nan,
                                           recovery_pct_stage_flagged=100*recovery))
                    for (s, c), (tol_bad, phys_bad, par_bad, centre, tol) in per_parameter.items():
                        if s != stage:
                            continue
                        other = np.logical_or.reduce([f[2] for (ss, cc), f in per_parameter.items() if ss == stage and cc != c]) if sum(ss == stage for ss, _ in per_parameter) > 1 else np.zeros(n, bool)
                        for policy in ["direct_reject", "flag_only"]:
                            parameters.append(dict(path_id=pid, policy=policy, stage=stage, parameter=c,
                                                   intermediate_tolerance_pct=100*tau, final_tolerance_pct=100*final_tau,
                                                   centre=centre, parameter_tolerance_pct=100*tol, n_particles=n,
                                                   tolerance_flag_pct=100*tol_bad.mean(), physical_flag_pct=100*phys_bad.mean(),
                                                   any_flag_pct=100*par_bad.mean(),
                                                   first_rejected_and_parameter_failed_pct_original=100*(newly & par_bad).mean() if policy == "direct_reject" else 100*(final_bad & par_bad).mean() if stage == "Electrochemical" else 0,
                                                   exclusively_rejected_by_parameter_pct_original=100*(newly & par_bad & ~other).mean() if policy == "direct_reject" else 100*(final_bad & par_bad & ~other).mean() if stage == "Electrochemical" else 0,
                                                   final_pass_among_parameter_flagged_pct=100*(par_bad & flagging).sum()/par_bad.sum() if par_bad.sum() else np.nan))
                    alive_before = alive_after
        ranking = pd.DataFrame(rankings)
        for _, ix in ranking.groupby(["intermediate_tolerance_pct", "final_tolerance_pct"]).groups.items():
            ranking.loc[ix, "rank_flagging"] = ranking.loc[ix, "flagging_yield_pct"].rank(method="min", ascending=False)
            ranking.loc[ix, "rank_direct"] = ranking.loc[ix, "direct_yield_pct"].rank(method="min", ascending=False)
        ranking = ranking.sort_values(["intermediate_tolerance_pct", "final_tolerance_pct", "rank_flagging", "rank_direct", "path_id"])
        self.save(ranking, report_prefix+"_ranking")
        self.save(pd.DataFrame(stages), report_prefix+"_scrap_stage")
        self.save(pd.DataFrame(parameters), report_prefix+"_scrap_parameter")
        self.physical_rules = pd.DataFrame(physical_rows)
        if save_physical_rules:
            self.save(self.physical_rules, report_prefix+"_scrap_physical_rules")
        self.ranking = ranking
        self.stage_scrap = pd.DataFrame(stages)
        self.parameter_scrap = pd.DataFrame(parameters)
        return ranking


class ScrapAnalyzer(ScrapPolicyMixin):
    def __init__(self,target,output_dir=None):
        self.config=SearchConfig()
        self.target=np.asarray(target,float)
        if self.target.shape!=(2,) or not np.isfinite(self.target).all() or np.any(self.target<=0):
            raise ValueError('Supply positive gravimetric and volumetric target capacities')
        self.output_dir=Path(output_dir) if output_dir else None

    def save(self,table,name):
        if self.output_dir:
            self.output_dir.mkdir(parents=True,exist_ok=True)
            table.to_csv(self.output_dir/(name+'.csv'),index=False)
        return table

def assess_scrap(particles,target,tolerance_pairs=((.02,.02),(.05,.05),(.1,.1)),output_dir=None):
    work=particles.copy()
    if 'path_id' not in work:work['path_id']=work['scenario_index']
    analyzer=ScrapAnalyzer(target,output_dir)
    analyzer.alternatives=work[['path_id']].drop_duplicates()
    analyzer.assess_scrap(work,tolerance_pairs=tolerance_pairs,verify_orchestration=False)
    return dict(ranking=analyzer.ranking,stages=analyzer.stage_scrap,parameters=analyzer.parameter_scrap,physical_rules=analyzer.physical_rules)
