## Description
Describe the changes made, including the motivation and context.

## Related Ticket
Fixes # (issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] New linking (a new preset in `configs/`, plus any new slot implementation)
- [ ] Change to a preset or a default (changes what an existing run name means)
- [ ] Documentation only
- [ ] Breaking change

## Presets affected
Which of `configs/*.env` does this change the model for (`cesm_numerical`, `jcm_numerical`, none)?

## How Has This Been Tested?
CI cannot run the model (no GPU, no CESM archive), so say what was run on the box.
- [ ] `scripts/validation/test_conservation.py`, `test_physics_math.py`, `test_configs.py` pass locally
- [ ] `run_prod.py --preset <name> --dry-run` for every affected preset resolves the intended values
- [ ] Smoke run from `runs/<tag>/` (preset, steps, and the budget closure line):
- [ ] Advection changes validated at `ADV_F32=1 ADV_CFL=0.5`

## Checklist
- [ ] Targets `main`, from a short-lived `feat/`, `fix/` or `docs/` branch
- [ ] A changed preset or default has a dated `MANIFEST.md` entry saying why
- [ ] New or moved submodule pins are exact commits, and `#@submodule` lines match the gitlinks
- [ ] No run output (`.npz`, `.png`, `.gif`) or anything outside `runs/` / `outputs/` committed
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
