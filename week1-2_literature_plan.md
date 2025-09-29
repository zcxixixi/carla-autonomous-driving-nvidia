# Week 1-2 Literature Review Checklist

## ? **Priority Reading List (Week 1-2)**

### **? Must Read This Week (7 papers)**

#### **Day 1-2: Safety & Planning**
- [ ] **Conditional Affordance Learning for Driving in Urban Environments** (Sauer et al., 2018)
  - Focus: How they handle safety in urban driving
  - Link: https://arxiv.org/abs/1806.06498
  - Key points to extract: Safety mechanisms, urban scenario handling

- [ ] **MultiNet: Real-time Joint Semantic Reasoning** (Teichmann et al., 2016)  
  - Focus: Multi-task learning architecture
  - Link: https://arxiv.org/abs/1612.07695
  - Key points: Multi-task loss design, architecture choices

#### **Day 3-4: Advanced End-to-End**
- [ ] **Exploring the Limitations of Behavior Cloning** (Codevilla et al., 2019)
  - Focus: Limitations of imitation learning
  - Link: https://arxiv.org/abs/1904.08980
  - Key points: Why pure imitation fails, safety issues

- [ ] **CARLA: An Open Urban Driving Simulator** (Dosovitskiy et al., 2017)
  - Focus: Simulation environment capabilities
  - Link: https://arxiv.org/abs/1711.03938
  - Key points: Evaluation metrics, scenario generation

#### **Day 5-6: Safety-Aware Systems**
- [ ] **Safe Control under Uncertainty with Probabilistic Signal Temporal Logic** (Sadigh et al., 2016)
  - Focus: Formal safety guarantees
  - Link: https://arxiv.org/abs/1605.04326
  - Key points: Safety formalization, uncertainty handling

- [ ] **Learning Robust, Real-Time, Reactive Robotic Movement** (Ratliff et al., 2018)
  - Focus: Reactive safety in robotics
  - Link: https://journals.sagepub.com/doi/10.1177/0278364918804654
  - Key points: Real-time safety responses

#### **Day 7: Survey & Synthesis**
- [ ] **Deep Learning for Multi-Modal Perception** (Liang et al., 2019)
  - Focus: Multi-modal fusion for safety
  - Link: https://arxiv.org/abs/1902.11365
  - Key points: Sensor fusion for safety-critical systems

### **? For Each Paper, Document:**

#### **Reading Template:**
```markdown
## Paper: [Title]
**Authors:** [Authors]
**Year:** [Year]
**Venue:** [Conference/Journal]

### Key Contributions:
1. 
2. 
3. 

### Methodology:
- Architecture:
- Training:
- Evaluation:

### Relevance to Our Work:
- What they do similar:
- What they miss:
- How we're different:

### Useful Technical Details:
- Implementation details:
- Datasets used:
- Metrics:

### Limitations/Future Work:
- 

### Our Innovation Positioning:
- 
```

## ? **Week Actions Checklist**

### **Monday (Today!)**
- [ ] Set up Zotero/Mendeley literature manager
- [ ] Download all 7 papers above
- [ ] Read first paper: Conditional Affordance Learning
- [ ] Fill out reading template
- [ ] Start comparison table

### **Tuesday**
- [ ] Read MultiNet paper
- [ ] Update comparison table
- [ ] Start drafting "Related Work" section outline

### **Wednesday**  
- [ ] Read Behavior Cloning limitations paper
- [ ] Identify gaps our method fills
- [ ] Update positioning statement

### **Thursday**
- [ ] Read CARLA evaluation paper
- [ ] Plan evaluation metrics for our work
- [ ] Design experimental scenarios

### **Friday**
- [ ] Read safety-aware control papers (2 papers)
- [ ] Document safety formalization approaches
- [ ] Compare with our safety learning approach

### **Weekend**
- [ ] Read multi-modal perception survey
- [ ] Synthesize all readings into comparison table
- [ ] Write first draft of Related Work section (500 words)

## ? **Comparison Table Template**

Create this table as you read:

| Paper | Year | Main Approach | Safety Mechanism | Limitations | Our Advantage |
|-------|------|---------------|------------------|-------------|---------------|
| NVIDIA | 2016 | End-to-end CNN | None | No safety awareness | Learn from accidents |
| ChauffeurNet | 2018 | Structured learning | Collision checking | Rule-based safety | Learned safety |
| MultiNet | 2016 | Multi-task | Semantic segmentation | No accident learning | Direct accident data |
| ... | ... | ... | ... | ... | ... |

## ? **Success Metrics for This Week**

By Sunday, you should have:
- [ ] 7 papers read and documented
- [ ] Complete comparison table (10+ methods)
- [ ] Related Work section first draft (500+ words)  
- [ ] Clear positioning statement for our approach
- [ ] List of evaluation metrics from literature

## ? **Pro Tips**

1. **Speed Reading Strategy:**
   - Abstract + Conclusion first (5 min)
   - Skim methodology (10 min)
   - Deep read relevant sections (15 min)
   - Total: 30 min per paper

2. **Note-Taking:**
   - Use consistent template
   - Focus on differentiation from our approach
   - Extract concrete technical details

3. **Don't Get Stuck:**
   - If a paper is too dense, skim and move on
   - Focus on understanding the landscape
   - Deep technical details can come later

## ? **Help Resources**

- **If papers are behind paywall:** Use Sci-Hub or university library
- **If concepts unclear:** YouTube explanations, Wikipedia
- **If overwhelmed:** Focus on abstracts and conclusions first

---

**Time Budget:** 3-4 hours per day this week
**Expected Output:** Solid foundation for Related Work section
**Next Week:** Start dataset construction while continuing literature review