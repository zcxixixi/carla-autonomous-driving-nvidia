# Paper Reading Notes: Conditional Affordance Learning

## Paper: Conditional Affordance Learning for Driving in Urban Environments
**Authors:** Axel Sauer, Nikolay Savinov, Andreas Geiger  
**Year:** 2018  
**Venue:** Conference on Robot Learning (CoRL) 2018  
**Citation:** arXiv:1806.06498 [cs.RO]  
**PDF Link:** https://arxiv.org/pdf/1806.06498  

---

## ? **Abstract Summary**

### **Problem Statement:**
- Most autonomous driving approaches fall into two categories:
  1. **Modular pipelines**: Build extensive environmental models
  2. **Imitation learning**: Map images directly to control outputs
- **Direct perception** (third paradigm) aims to combine advantages of both
- **Limitation**: Existing direct perception approaches only work for simple highway situations

### **Key Innovation:**
- **First direct perception approach** for complex urban environments
- Handles **traffic lights, speed signs, intersections** using only image-level labels
- Maps video input to intermediate representations for urban navigation
- **68% improvement** over state-of-the-art reinforcement and imitation learning on CARLA

---

## ? **Methodology**

### **Core Approach: Conditional Affordance Learning**
```
Video Input ¡ú Neural Network ¡ú Affordance Representations ¡ú Control Outputs
                                      ¡ü
                               High-level commands
```

### **Key Technical Components:**
1. **Affordance Representations**: Low-dimensional intermediate features
2. **Conditional Learning**: Uses high-level directional inputs (turn left/right/straight)
3. **Multi-task Learning**: Simultaneously learns multiple driving affordances
4. **Image-level Supervision**: Only requires labels on images, not full 3D scene understanding

### **What are "Affordances"?**
- **Definition**: Action possibilities that environment offers to an agent
- **Examples in driving**:
  - Distance to vehicle ahead
  - Distance to lane boundaries  
  - Traffic light state
  - Speed limit information
  - Intersection navigation cues

---

## ? **Key Contributions**

1. **First urban direct perception system**: Extends beyond highways to complex urban scenarios
2. **Traffic infrastructure handling**: First to handle traffic lights and speed signs with image-only labels
3. **Performance breakthrough**: 68% improvement in goal-directed navigation on CARLA
4. **Smooth behavior**: Achieves smooth car-following and reduces traffic accidents
5. **Practical supervision**: Only requires image-level labels (much easier to collect than full 3D annotations)

---

## ? **Relevance to Our Accident-Aware Research**

### **What They Do Similar:**
- Use **intermediate representations** instead of direct image-to-control mapping
- Work in **CARLA simulation environment**
- Focus on **urban driving scenarios** with complex interactions
- Use **multi-task learning** approach

### **What They Miss (Our Opportunity):**
- **No accident learning**: Their safety comes from rule-based traffic light/sign detection
- **No failure case learning**: They don't learn from negative driving examples
- **Static safety rules**: Safety is hardcoded rather than learned from experience
- **No accident-aware representations**: Their affordances don't include accident risk assessment

### **How We're Different:**
- **Learn from accident data**: Our approach learns safety from failed driving examples
- **Dynamic safety learning**: Safety awareness emerges from data, not rules
- **Accident-specific affordances**: We learn representations that include accident risk
- **Two-stage training**: Normal driving + accident-aware fine-tuning

---

## ? **Useful Technical Details**

### **Architecture Details:**
- **Input**: Video sequences (multiple frames)
- **Backbone**: Convolutional neural network
- **Output**: Multiple affordance predictions
- **Conditioning**: High-level navigation commands

### **Training Strategy:**
- **Supervised learning** on human driving data
- **Multi-task loss function** for different affordances
- **Data augmentation** for robustness

### **Evaluation Metrics:**
- **Success rate** in reaching goals
- **Number of infractions** (traffic violations)
- **Collision rate**
- **Smooth driving metrics**

### **Dataset:**
- **CARLA simulation** with diverse urban scenarios
- **Image-level labels** for traffic lights, signs, etc.
- **Human demonstration data** for driving behavior

---

## ?? **Limitations/Future Work**

### **Current Limitations:**
1. **Still simulation-only**: Not tested on real-world data
2. **Limited safety modeling**: Safety rules are hardcoded
3. **No adversarial scenarios**: Doesn't handle deliberate failure cases
4. **Static affordances**: Affordance definitions are predetermined

### **Future Work Suggestions:**
- Real-world validation
- More complex urban scenarios
- **Dynamic safety learning** (our opportunity!)
- Integration with other sensing modalities

---

## ? **Our Innovation Positioning**

### **How We Build Upon This Work:**
1. **Use their affordance framework** as foundation
2. **Add accident-aware affordances**:
   - Risk assessment score
   - Accident probability prediction
   - Safety-critical situation detection
3. **Extend their training methodology**:
   - Stage 1: Learn normal affordances (like their approach)
   - Stage 2: Learn accident-aware affordances from failure data
4. **Combine their urban capabilities with our safety learning**

### **Our Unique Value Proposition:**
```
Conditional Affordance Learning + Accident Learning = 
Accident-Aware Conditional Affordance Learning
```

### **Technical Differentiation:**
- **Their affordances**: Distance, speed, traffic signals
- **Our additional affordances**: Accident risk, safety margin, failure prediction
- **Their training**: Positive examples only
- **Our training**: Positive + negative (accident) examples

---

## ? **Integration with Our Approach**

### **What We Can Adopt:**
- ? **Affordance representation framework**
- ? **Multi-task learning architecture** 
- ? **CARLA evaluation methodology**
- ? **Image-level supervision approach**

### **What We Need to Extend:**
- ? **Add accident-aware affordances**
- ? **Implement two-stage training**
- ? **Create accident scenario datasets**
- ? **Design safety-specific loss functions**

### **Our Enhanced Architecture:**
```
Video Input ¡ú Enhanced Neural Network ¡ú {Normal Affordances + Safety Affordances} ¡ú Safe Control
                                              ¡ü                    ¡ü
                                    High-level commands    Accident Learning
```

---

## ? **Action Items for Our Research**

### **Immediate (This Week):**
- [ ] Download and read full PDF
- [ ] Study their affordance definitions in detail
- [ ] Understand their multi-task loss function
- [ ] Compare with NVIDIA end-to-end approach

### **Implementation (Next Week):**
- [ ] Adapt their affordance framework to our accident-aware system
- [ ] Design safety-specific affordances
- [ ] Plan how to integrate accident learning with their approach

### **Evaluation (Later):**
- [ ] Use their CARLA evaluation scenarios
- [ ] Compare our safety performance against their baseline
- [ ] Show improvement in accident reduction metrics

---

## ? **Key Insights for Literature Review**

### **For Related Work Section:**
1. **Position as evolution**: "Building upon conditional affordance learning..."
2. **Highlight gap**: "While effective for normal driving, lacks accident awareness..."
3. **Show complementarity**: "Our approach extends affordance learning to safety-critical scenarios..."

### **For Methodology Section:**
1. **Adopt their intermediate representation philosophy**
2. **Extend their multi-task learning to include safety tasks**
3. **Use their CARLA evaluation framework as baseline**

### **For Innovation Claims:**
1. **First to combine affordance learning with accident learning**
2. **Novel safety-aware affordances**
3. **Two-stage training methodology for safety**

---

**? Overall Assessment: HIGHLY RELEVANT**
- Direct technical foundation for our approach
- Strong baseline to build upon
- Clear differentiation opportunity through accident learning
- Excellent evaluation framework we can adopt

**Next Paper Priority: MultiNet (multi-task architecture details)**

## ?????????????

????????????????????????????????????????????

| ?? | ?? | ?? / ?? | ????????? |
|---:|---|---|---|
| 1 | ??/?? | ????????????/??/?? | ??????????? |
| 2 | ????/??? | ????????????1-2 ??? | ????????????? |
| 3 | ???? | ???????/??/??/??/??????? | ?????????????? |
| 4 | ????? | ???????????????????????? | ????????? schema?affordance schema? |
| 5 | ??/???? | ??????????????????? | ?????????????? |
| 6 | ???? | ?????????????????PID/LQR/??? | ????????????????? |
| 7 | ????? | ????????????????? | ????????????? |
| 8 | ??????? | ???????????? | ???????? failure learning????? |

????????????????????????????? `papers/` ???????????????? `notes/paper_index.md`???????????

## Eight-item Reading Checklist (Template)

Below is an English version of the eight-item checklist that you can use for every paper to structure findings into engineering tasks and experiments.

| # | Item | Description / Key points | Direct relevance to our project |
|---:|---|---|---|
| 1 | Goal / Problem | The core problem the paper addresses; inputs / outputs / constraints | Clarify reproduction target and evaluation task |:To automaticly drive in the environment of urban
| 2 | Core idea / Innovation | The main contribution compared to prior work, in 1-2 sentences | Determines which modules to reproduce or adapt |
| 3 | System architecture | Module breakdown (perception / decision / control / training / evaluation) and data flow | Guides engineering interfaces and modular implementation |
| 4 | Data & labels | Required data types, label formats, collection methods, rare-sample handling | Drives data collection scripts and affordance schema |
| 5 | Training / optimization details | Loss functions, weighting, temporal modeling, augmentation | Key hyperparameters and tricks for reproduction |
| 6 | Control strategy | End-to-end or rule-based control; controller details (PID/LQR/model) | Determines deployment safety constraints and control implementation |
| 7 | Experiments & evaluation | Benchmarks, metrics, baselines, scenario configuration | Plans reproduction experiments and evaluation scripts |
| 8 | Limitations & future work | Shortcomings the authors mention and suggested extensions | Informs next steps (e.g., adding failure learning) |

How to use: Fill this checklist for each paper you read and save the results under `papers/` (or collect them in `notes/paper_index.md`) for quick comparison and retrieval.

---

## Conditional Affordance — Completed 8?Item Checklist

| # | Item | Summary (for this paper) | Direct relevance to our accident-aware project |
|---:|---|---|---|
| 1 | Goal / Problem | Learn to drive in complex urban environments by predicting low-dimensional affordances from video, conditioned on high-level navigation commands (left/right/straight). Inputs: multi-frame RGB; Outputs: affordance vector (distances, signals, flags). | Clear reproduction target: implement perception?affordances pipeline; defines required inputs/outputs for data collection and model IO. |
| 2 | Core idea / Innovation | Conditional Affordance Learning: combine direct perception with conditional multi-task heads so a single model predicts multiple affordances that feed a rule-based controller; handles traffic lights, signs, intersections with image-level labels. | Use their affordance representation and conditional heads as baseline; core idea motivates adding a risk-affordance head for failure learning. |
| 3 | System architecture | CNN backbone (multi-frame input) ? conditional injection of high-level command ? multiple task-specific heads (regression & classification) ? affordances ? rule-based controller (lateral PID, distance-based longitudinal). | Implement modular pipeline: separate perception module (model + heads) and controller module so we can add risk head and safe-switching without touching control logic. |
| 4 | Data & labels | CARLA-simulated frame-level labels: front_vehicle_distance (m, reg), left_lane_dist (m), right_lane_dist (m), traffic_light_state (enum), speed_limit (enum), intersection_flag (bool), high_level_command (enum). Authors use image-level supervision only; no 3D scene reconstruction required. | Use this exact schema as `data/affordance_schema.json`; add `risk_score` and `time_to_collision` fields for accident-aware data collection (can be filled from simulation telemetry or annotated accident records). |
| 5 | Training / optimization details | Supervised multi-task learning: MSE for regressions, cross-entropy for classification; weighted sum of task losses; use multiple stacked frames or temporal modules for context; data augmentation for robustness. | Reuse loss composition and temporal input strategy; record task weights and experiment with focal loss or class weighting for rare accident labels when adding risk head. |
| 6 | Control strategy | Not end-to-end. Use a rule-based controller: lateral control via PID on lane offsets, longitudinal control via distance-based policy (target speed, emergency braking at traffic lights/obstacles). | Keep rules as safety net; implement controller interface to accept `risk_score`—when risk exceeds threshold, switch to conservative controller (reduce speed / increase gap / emergency brake). |
| 7 | Experiments & evaluation | Evaluate on CARLA goal-directed navigation tasks across multiple towns and traffic densities. Metrics: success rate (goal reached), collision count, traffic infractions, and smoothness measures. Compare against imitation and RL baselines. | Reuse their evaluation protocol for direct comparison; add new metrics for early-warning quality (ROC/AUC of `risk_score` predicting accidents) and intervention effectiveness. |
| 8 | Limitations & future work | Evaluated in simulation only; safety logic is hand-coded; no learning from failure cases or adversarial scenarios; affordances are manually defined and may miss safety-relevant signals. | Provides motivation for our contributions: add accident/failure learning, learn risk-aware affordances, and validate in richer/rare-event scenarios. |

Notes / Quick actionables:
- Create `data/affordance_schema.json` with fields listed in item 4 and add `risk_score` (0-1) and `time_to_collision` (s).
- Implement perception module with conditional heads in `algorithms/` and a controller wrapper that listens to `risk_score` to switch modes.
- Evaluation: replicate CARLA benchmarks from the paper, then run ablation: baseline CAL vs CAL+RiskHead.