# On the Faithfulness of Vision Transformer Explanations

![Gradient-Based Explanation](assets/gradcam.gif)
---

1. [Introduction](#1-introduction)
   - [Vision Transformers](#11-vision-transformers-and-their-rise)
   - [Explainable AI in Vision Transformers](#12-explainable-ai-and-vision-transformers)
   - [Paper Context and Research Problem](#13-paper-context-and-research-problem)
   - [Project Goals](#14-my-goal)
   - [Visuals and Examples](#15-visuals-and-examples)
2. [Methodology](#2-the-method-and-our-interpretation)
   - [The Paper’s Approach](#21-methodology-the-papers-approach)
   - [My Interpretation](#22-our-interpretation-evaluating-saco-and-suggestions-for-improvement)
3. [Experiments and Results](#3-experiments-and-results)
   - [Experimental Setup](#31-experimental-setup)
   - [Results](#32-results)
4. [Conclusion](#4-conclusion)
5. [References](#5-references)
6. [Contact](#6-contact)


---
## 1. Introduction
---

### 1.1 Vision Transformers and Their Rise

Vision Transformers (ViTs) have introduced a transformative approach to computer vision by adapting the transformer architecture, originally designed for natural language processing, to visual tasks. Unlike convolutional neural networks (CNNs), which build features hierarchically through localized operations, ViTs divide an image into small fixed-size patches. These patches are then embedded into vectors and processed as sequences using the transformer’s self-attention mechanism [1].

A key strength of ViTs lies in their ability to model global relationships across an image from the very first layer. This contrasts with CNNs, which rely on progressively larger receptive fields to capture global context. By using self-attention, ViTs effectively analyze long-range dependencies and contextual information, making them particularly well-suited for tasks requiring a holistic understanding of images, such as image classification, object detection, and segmentation [2, 3].

Critical issue with ViTs is their interpretability. The self-attention mechanism is complex, making ViTs behave as "black-box" models. Understanding **why** and **how** ViTs make specific predictions is essential, especially in safety-critical areas like medical diagnosis or autonomous driving, where trust in the model's decisions is vital [3]. This need for interpretability has driven research into Explainable AI (XAI) techniques specifically tailored to Vision Transformers.

---

### 1.2 Explainable AI and Vision Transformers

Explainable AI (XAI) aims to provide insights into how machine learning models, particularly deep neural networks, make predictions. By improving transparency and interpretability, XAI fosters trust in AI systems, especially in critical applications like healthcare, finance, and autonomous driving. In the context of Vision Transformers (ViTs), explainability is crucial to understanding the internal mechanisms of these complex models, often regarded as "black-boxes."

There are three primary categories of explanation methods applied to Vision Transformers: gradient-based methods, attribution-based methods, and attention-based methods. Each approach provides a unique perspective on the decision-making process of these models.

#### Gradient-Based Methods
Gradient-based methods analyze the gradient of the output with respect to the input to identify important features for a model's prediction. We select two state-of-the-art gradient-based methods in our experiments:
- **Integrated Gradients (IG)** [4]: A method that attributes the importance of each input feature by integrating gradients along a path from a baseline to the input.
- **Grad-CAM** [5]: Originally designed for CNNs, Grad-CAM generates visual explanations by leveraging gradients of the target class with respect to intermediate feature maps. Our implementation follows prior work on Vision Transformer interpretability [3].

#### Attribution-Based Methods
Attribution-based methods explicitly model the information flow inside the network to understand how different components contribute to predictions. These methods offer a more direct interpretation of model behavior compared to gradient-based methods. In our experiments, we include:
- **Layer-wise Relevance Propagation (LRP)** [6]: Decomposes predictions to attribute relevance scores back to input features.
- **Partial LRP** [7]: A variant of LRP tailored for more specific relevance assignments.
- **Conservative LRP** [8]: Focuses on providing conservative and consistent attributions.
- **Transformer Attribution**: Designed specifically for the ViT architecture to trace information flow.

#### Attention-Based Methods
Attention-based methods leverage the self-attention mechanism of ViTs to provide explanations. These methods focus on interpreting the attention weights that determine how image patches interact within the model. In our experiments, we employ four variants:
- **Raw Attention** [9]: Visualizes raw attention weights directly from the transformer layers.
- **Attention Rollout** [10]: Aggregates attention weights across all layers to provide a global explanation.
- **Transformer-MM** [11]: A method designed to extend interpretability by focusing on multi-modal transformers.
- **ATTCAT** [12]: Employs category-specific attention visualization for a more fine-grained explanation.

Each of these methods contributes to a comprehensive analysis of Vision Transformers, helping to demystify their decision-making processes while highlighting their strengths and limitations.

---
### 1.3 Paper Context and Research Problem

The rapid adoption of Vision Transformers (ViTs) in computer vision necessitates reliable evaluation metrics for post-hoc explanation methods. These methods aim to provide human-understandable heatmaps by assigning salience scores to input pixels, with the expectation that higher scores correspond to greater influence on the model's predictions [3, 5]. However, existing metrics exhibit significant limitations, raising concerns about their ability to evaluate the faithfulness of these explanations rigorously.

#### Challenges in Evaluating Faithfulness
Faithfulness, defined as the degree to which explanations reflect the true decision-making processes of a model, is crucial for building trust in explanations [6, 8]. Current evaluation metrics, particularly those based on **cumulative perturbation**, fall short in key areas:
1. **Conflated Impacts**: Cumulative perturbation sequentially removes groups of pixels with high salience scores, conflating their individual contributions. For instance, the influence of the top 0–10% salient pixels cannot be disentangled from the next 90–100%, making it impossible to evaluate their impacts independently [3, 6].
2. **Lack of Granularity**: Existing metrics do not leverage the full distribution of salience scores. They rely on the ranking of salience values but fail to quantify differences between scores, overlooking whether higher-scored pixels truly exert proportionally greater influence on predictions [5, 7].
3. **Failure to Distinguish Methods**: Alarmingly, current metrics often fail to differentiate between advanced explanation methods and Random Attribution, highlighting their inability to validate the core assumptions of faithfulness [6].



Given these challenges, the paper addresses the following key research problems:
1. How can we develop a robust metric to evaluate the faithfulness of ViT explanations?
2. How can we quantify the contributions of individual pixel subsets while accounting for differences in salience score magnitudes?
3. How can we ensure that faithfulness metrics reliably distinguish between advanced explanation methods and random baselines?

#### Contributions of SaCo
To address these questions, the paper introduces the **Salience-guided Faithfulness Coefficient (SaCo)**, a novel evaluation framework designed to:
1. Analyze the statistical relationships between salience scores and their actual impact on model predictions.
2. Evaluate the model’s response to distinct pixel groups, explicitly capturing the expected disparities in their contributions.
3. Provide a robust benchmark for distinguishing meaningful explanations from random attribution methods.

SaCo’s evaluation across multiple datasets and ViT models reveals that most existing explanation methods fail to meet the core assumption of faithfulness. These findings underscore the need for more rigorous evaluation frameworks and offer insights into design factors—such as gradient information and aggregation strategies—that can improve the faithfulness of ViT explanation methods.

By addressing these gaps, SaCo establishes a pathway for advancing the interpretability of ViTs, ensuring explanations not only align with human intuition but also faithfully reflect the model’s decision-making processes.

---
### 1.4 My Goal
The goal of this project is to
- **Reproduce the SaCo metric** as described in the paper to verify its reproducibility and reliability.
- **Explore its application** across different datasets and model architectures to test its generalizability.

---

### References
1. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. Khan, S., et al. (2022). "Transformers in Vision: A Survey." [ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3505244)
3. Chefer, H., et al. (2021). "Transformer Interpretability Beyond Attention Visualization." [arXiv:2012.09838](https://arxiv.org/abs/2012.09838)
4. Sundararajan, M., et al. (2017). "Axiomatic Attribution for Deep Networks." [ICML](https://arxiv.org/abs/1703.01365)
5. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." [ICCV](https://arxiv.org/abs/1610.02391)
6. Bach, S., et al. (2015). "Pixel-wise Explanations for Non-linear Classifiers as a Function of Input Features." [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
7. Montavon, G., et al. (2019). "Layer-wise Relevance Propagation: An Overview." [arXiv:1902.05649](https://arxiv.org/abs/1902.05649)
8. Samek, W., et al. (2021). "Towards Explainable AI: Layer-wise Relevance Propagation." [arXiv:2021.00508](https://arxiv.org/abs/2021.00508)
9. Vaswani, A., et al. (2017). "Attention is All You Need." [NeurIPS](https://arxiv.org/abs/1706.03762)
10. Abnar, S., and Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)
11. Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
12. Zhao, X., et al. (2021). "ATTCAT: Attention Category Visualization for Transformers." [arXiv:2021.03017](https://arxiv.org/abs/2021.03017)










