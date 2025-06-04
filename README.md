# Awesome-Multimodal-Sentiment-Analysis
✨✨Latest Advances on Multimodal Sentiment Analysis

# 🎨 Introduction 
Multimodal Sentiment Analysis is a computational task that aims to detect and interpret sentiment or emotional states by integrating heterogeneous data from multiple modalities, such as language (text or speech), vision (facial expressions or visual scenes), and audio (tone, pitch, prosody). The objective is to leverage the complementary and synergistic cues from different modalities to achieve a more robust and nuanced understanding of human affective expressions.

---

# 📕 Table of Contents
- [🌷 Scene Graph Datasets](#-datasets)
- [🍕 Scene Graph Generation](#-scene-graph-generation)
  - [2D (Image) Scene Graph Generation](#2d-image-scene-graph-generation)
  - [Panoptic Scene Graph Generation](#panoptic-scene-graph-generation)
  - [Spatio-Temporal (Video) Scene Graph Generation](#spatio-temporal-video-scene-graph-generation)
- [🥝 Scene Graph Application](#-scene-graph-application)
  - [Image Retrieval](#image-retrieval)
  - [Image/Video Caption](#imagevideo-caption)
  - [2D Image Generation](#2d-image-generation)
- [🤶 Evaluation Metrics](#evaluation-metrics)
- [🐱‍🚀 Miscellaneous](#miscellaneous)
  - [Toolkit](#toolkit)
  - [Workshop](#workshop)
  - [Survey](#survey)
  - [Insteresting Works](#insteresting-works)
- [⭐️ Star History](#️-star-history)


---

# 🌷 Scene Graph Datasets
<p align="center">

| Dataset |  Modality  |   Obj. Class  | BBox | Rela. Class | Triplets | Instances | 
|:--------:|:--------:|:--------:| :--------:|  :--------:|  :--------:|  :--------:|
| [Visual Phrase](https://vision.cs.uiuc.edu/phrasal/) | Image | 8 | 3,271 | 9 | 1,796 | 2,769 |
| [Scene Graph](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf) | Image | 266 | 69,009 | 68 | 109,535 | 5,000 |
| [VRD](https://cs.stanford.edu/people/ranjaykrishna/vrd/)  | Image | 100 | - | 70 | 37,993 | 5,000 |
| [Open Images v7](https://storage.googleapis.com/openimages/web/index.html)  | Image | 600 | 3,290,070 | 31 | 374,768 | 9,178,275 |
| [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | Image | 5,996 | 3,843,636 | 1,014 | 2,347,187 | 108,077 | 
</p>


---

<!-- CVPR-8A2BE2 -->
<!-- WACV-6a5acd -->
<!-- NIPS-CD5C5C -->
<!-- ICML-FF7F50 -->
<!-- ICCV-00CED1 -->
<!-- ECCV-1e90ff -->
<!-- TPAMI-BC8F8F -->
<!-- IJCAI-228b22 -->
<!-- AAAI-c71585 -->
<!-- arXiv-b22222 -->
<!-- ACL-191970 -->
<!-- TPAMI-ffa07a -->


# 🍕 Scene Graph Generation

## 2D (Image) Scene Graph Generation

There are three subtasks:
- `Predicate classification`: given ground-truth labels and bounding boxes of object pairs, predict the predicate label.
- `Scene graph classification`: joint classification of predicate labels and the objects' category given the grounding bounding boxes.
- `Scene graph detection`: detect the objects and their categories, and predict the predicate between object pairs.

### LLM-based 

+ [**A fine-grained modal label-based multi-stage network for multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S0957417423002221) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**A Fine-Grained Tri-Modal Interaction Model for Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/10447872/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**A Multimodal Sentiment Analysis Model Enhanced with Non-verbal Information and Contrastive Learning**](https://jeit.ac.cn/en/article/doi/10.11999/JEIT231274) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**A NOVEL MULTIMODAL SENTIMENT ANALYSIS MODEL BASED ON GATED FUSION AND MULTI-TASK LEARNING**](https://ieeexplore.ieee.org/abstract/document/10446040/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality**](https://aclanthology.org/2020.acl-main.343/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**All-modalities-in-One BERT for multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S1566253522002329) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations**](https://ieeexplore.ieee.org/document/10109845) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Bi-Direction Attention Based Fusion Network for Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/9932611/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**CLMLF-A Contrastive Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection**](https://arxiv.org/abs/2204.05515) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Contrastive Feature Decomposition for Multimodal Sentiment Analysis**](https://aclanthology.org/2023.acl-long.421/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Contrastive Knowledge Injection for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2306.15796) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Context-Based Adaptive Multimodal Fusion Network for Continuous Frame-Level Sentiment Prediction**](https://ieeexplore.ieee.org/abstract/document/10271721/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Disentanglement Translation Network for multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S1566253523003470) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**Hierarchical Feature Fusion Network with Local and Global Perspectives for Multimodal Affective Computing**](https://aclanthology.org/P19-1046/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Multimodal Phased Transformer for Sentiment Analysis**](https://aclanthology.org/2021.emnlp-main.189/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**GUIDED CIRCULAR DECOMPOSITION AND CROSS-MODAL RECOMBINATION FOR MULTIMODAL SENTIMENT ANALYSIS**](https://ieeexplore.ieee.org/abstract/document/10446166/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Hybrid Contrastive Learning of Tri-Modal Representation for Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/9767560/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**An Extended Multimodal Adaptation Gate for Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/9746536/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Interactive Conversational Memory Network for Multimodal Emotion Detection**](https://aclanthology.org/D18-1280/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2109.00412) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Integrating Multimodal Information in Large Pretrained Transformers**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8005298/) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Inter-Intra Modal Representation Augmentation With Trimodal Collaborative Disentanglement Network for Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/10089492/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis**](https://dl.acm.org/doi/abs/10.1145/3394171.3413678) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**Mitigating Inconsistencies in Multimodal Sentiment Analysis under Uncertain Missing Modalities**](https://aclanthology.org/2022.emnlp-main.189/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Modality to Modality Translation An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion**](https://aaai.org/ojs/index.php/AAAI/article/view/5347) [![Paper](https://img.shields.io/badge/AAAI25-8B0000)]()

+ [**Modality translation-based multimodal sentiment analysis under uncertain missing modalities**](https://www.sciencedirect.com/science/article/pii/S1566253523002890) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**Hierarchical Graph Contrastive Learning for Multimodal Sentiment Analysis**](https://repository.essex.ac.uk/34855/) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos**](https://arxiv.org/abs/1606.06259) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Modal-Temporal Attention Graph for Unaligned Human Multimodal Language Sequences**](https://arxiv.org/abs/2010.11985) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**MULTI-CHANNEL ATTENTIVE GRAPH CONVOLUTIONAL NETWORK WITH SENTIMENT FUSION FOR MULTIMODAL SENTIMENT ANALYSIS**](https://ieeexplore.ieee.org/abstract/document/9747542/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multi-Channel Weight-Sharing Autoencoder Based on Cascade Multi-Head Attention for Multimodal Emotion Recognition**](https://ieeexplore.ieee.org/abstract/document/9693238/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**MULTI-GRAINED MULTIMODAL INTERACTION NETWORK FOR SENTIMENT ANALYSIS**](https://ieeexplore.ieee.org/abstract/document/10446351/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/8215597/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multimodal Contrastive Learning via Uni-Modal Coding and Cross-Modal Prediction for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2210.14556) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Multimodal Language Analysis with Recurrent Multistage Fusion**](https://arxiv.org/abs/1808.03920) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Multimodal Multi-loss Fusion Network for Sentiment Analysis**](https://aclanthology.org/2024.naacl-long.197/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Multimodal Phased Transformer for Sentiment Analysis**](https://aclanthology.org/2021.emnlp-main.189/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Multimodal Representations Learning Based on Mutual Information Maximization and Minimization and Identity Embedding for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2201.03969) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Multimodal Sentiment Analysis Based on Interactive Transformer and Soft Mapping**](https://onlinelibrary.wiley.com/doi/abs/10.1155/2022/6243347) [![Paper](https://img.shields.io/badge/Wiley25-9370DB)]()

+ [**Multimodal Sentiment Analysis with Word-Level Fusion and Reinforcement Learning**](https://dl.acm.org/doi/abs/10.1145/3136755.3136801) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**Multimodal Sentiment Analysis: A Survey and Comparison**](https://www.igi-global.com/chapter/multimodal-sentiment-analysis/308579) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**A systematic review of history, datasets, multimodal fusion methods, applications, challenges and future directions**](https://www.sciencedirect.com/science/article/pii/S1566253522001634) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**Multimodal Sentiment Detection Based on Multi-channel Graph Neural Networks**](https://aclanthology.org/2021.acl-long.28/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Multimodal Affective Computing With Dense Fusion Transformer for Inter- and Intra-Modality Interactions**](Multimodal_Affective_Computing_With_Dense_Fusion_Transformer_for_Inter-_and_Intra-Modality_Interactions) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**TEDT: Transformer-Based Encoding–Decoding Translation Network for Multimodal Sentiment Analysis**](https://link.springer.com/article/10.1007/s12559-022-10073-9) [![Paper](https://img.shields.io/badge/Springer25-20B2AA)]()

+ [**Improving Multimodal Sentiment Analysis via Multi-Scale Fusion of Locally Descriptors**](https://arxiv.org/abs/2112.01368) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Self-adaptive Context and Modal-interaction Modeling For Multimodal Emotion Recognition**](https://aclanthology.org/2023.findings-acl.390/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Sentiment Knowledge Enhanced Self-supervised Learning for Multimodal Sentiment Analysis**](https://aclanthology.org/2023.findings-acl.821/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Sentiment Word Aware Multimodal Refinement for Multimodal Sentiment Analysis with ASR Errors**](https://arxiv.org/abs/2203.00257) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Shared and Private Information Learning in Multimodal Sentiment Analysis with Deep Modal Alignment and Self-supervised Multi-Task Learning**](https://arxiv.org/abs/2305.08473) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Sentiment Knowledge Enhanced Attention Fusion Network for multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S1566253523002749) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**Sentimental Words Aware Fusion Network for Multimodal Sentiment Analysis**](https://aclanthology.org/2020.coling-main.93/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Tackling Modality Heterogeneity with Multi-View Calibration Network for Multimodal Sentiment Detection**](https://aclanthology.org/2023.acl-long.287/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Tag-assisted Multimodal Sentiment Analysis under Uncertain**](https://dl.acm.org/doi/abs/10.1145/3477495.3532064) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**Targeted Multimodal Sentiment Classifcation Based on Coarse-to-Fine Grained Image-Target Matching**](https://www.ijcai.org/proceedings/2022/0622.pdf) [![Paper](https://img.shields.io/badge/IJCAI25-DC143C)]()

+ [**Text-oriented Cross Attention Network for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2404.04545) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Text-centered fusion network with crossmodal attention for multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S0950705123002526) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**Tensor Fusion Network for Multimodal Sentiment Analysis**](https://arxiv.org/abs/1707.07250) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**A text enhanced transformer fusion network for multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S0031320322007385) [![Paper](https://img.shields.io/badge/Elsevier25-32CD32)]()

+ [**The Multimodal Sentiment Analysis in Car Reviews (MuSe-CaR) Dataset: Collection, Insights and Improvements**](https://ieeexplore.ieee.org/abstract/document/9484711/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**A New Token Mixup Multimodal Data Augmentation for Multimodal Sentiment Analysis**](https://dl.acm.org/doi/abs/10.1145/3543507.3583406) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**two-stage contrastive learning and feature hierarchical fusion network for multimodal sentiment analysis**](https://link.springer.com/article/10.1007/s00521-024-09634-w) [![Paper](https://img.shields.io/badge/Springer25-20B2AA)]()

+ [**Modulating Unimodal and Cross-modal Dynamics for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2111.08451) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**AAAI21 Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis**](https://ojs.aaai.org/index.php/AAAI/article/view/17289) [![Paper](https://img.shields.io/badge/AAAI25-8B0000)]()

+ [**Multimodal Transformer for Unaligned Multimodal Language Sequences**](https://arxiv.org/abs/1906.00295) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Hierarchical Learning for Multimodal Sentiment Analysis Using Coupled-Translation Fusion Network**](https://aclanthology.org/2021.acl-long.412/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**A Multimodal Language Dataset for Spanish, Portuguese, German and French**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8106386/) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**EMNLP21 Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2109.00412) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**MM21 Transformer-based Feature Reconstruction Network for Robust Multimodal Sentiment Analysis**](https://dl.acm.org/doi/abs/10.1145/3474085.3475585) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()
