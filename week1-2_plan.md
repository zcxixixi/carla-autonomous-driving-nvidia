# Week 1-2 Plan ¡ª Accident-Aware Conditional Affordance Learning

Objective: Implement a reproducible baseline of Conditional Affordance Learning (CAL) in CARLA, add a risk-prediction affordance, collect data, and run three baseline experiments.

Week 1 (Setup & Data)
- Day 1: Create `data/affordance_schema.json` and verify structure. (Done)
- Day 2: Run the `algorithms/collect_research_data.py` script in a short (5-minute) run to validate sensor saves and metadata writing.
- Day 3-4: Extend data collection to include several scenarios and intentionally create near-collision cases for risk labels.
- Day 5: Inspect collected data and produce a small dataset (¡Ö500-2000 frames) for quick model prototyping.

Week 2 (Model & Experiments)
- Day 6-7: Implement and run a minimal training loop using `algorithms/cal_model.py` (placeholder model) on the collected dataset.
- Day 8: Implement controller wrapper (`algorithms/controller.py`) and a small evaluation script to replay recorded sequences and compare control actions.
- Day 9-10: Run three experiments:
  1. Baseline CAL (no risk head)
  2. CAL + RiskHead (risk_score predicted; controller switches on high risk)
  3. Residual controller: rule-based controller + small network to predict residual control
- Day 11-12: Collect results, compute metrics (success rate, collisions per km, risk_score ROC/AUC), and prepare short report.

Quick Acceptance Criteria
- Schema file exists and is used by the collector.  
- Collector outputs images and per-frame JSON metadata.  
- A minimal model can load a batch of images and produce affordance outputs.  
- Controller can read affordances and produce control outputs; risk switching causes observable behavior change.

Notes
- Use synchronous mode for deterministic data collection.  
- When creating rare events, script scenarios that generate sudden braking or pedestrian crossings.  
- Keep experiments small and reproducible (use seeds and document CARLA map/time/weather settings).
