# Legacy analysis scripts (paper v3)

`extra_analysis_v3.py` and `make_paper_figures_v3.py` are the original scripts
behind the extra checks and figures of paper v3. They are kept for the record
and are not part of the current pipeline.

Use `../extra_analysis.py` and `../make_paper_figures.py` instead. They
compute the same quantities with the same method, and also:

- break ties at random in risk-coverage curves. ForgeHLS lists the Pareto
  variants of a kernel early in its row order, so the stable sort used here
  lets row order influence tied scores (see `../row_order_audit.py`);
- use a 5%–100% coverage grid (here 10%–100%);
- add the feature-collision floor and between-kernel variance share;
- are used by `../pareto_dse.py` and by `paper/build_paper_v4.py`.

The v3 scripts write to `results/extra/`; that folder is not produced by the
current pipeline.
