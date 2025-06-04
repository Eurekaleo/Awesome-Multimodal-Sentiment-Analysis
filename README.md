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

### Multimodal Sentiment Analysis (MSA) 

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
  

#### Multimodal Emotion Recognition in Conversation (MERC)

+ [**A Contextualized Real-Time Multimodal Emotion Recognition for Conversational Agents using Graph Convolutional Networks in Reinforcement Learning**](https://arxiv.org/abs/2310.18363) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**A Facial Expression-Aware Multimodal Multi-task Learning Framework for Emotion Recognition in Multi-party Conversations**](https://aclanthology.org/2023.acl-long.861/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**A Multitask learning model for multimodal sarcasm, sentiment and emotion recognition in conversations**](https://www.sciencedirect.com/science/article/pii/S1566253523000040) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**A novel spatio-temporal convolutional neural framework for multimodal emotion recognition**](https://www.sciencedirect.com/science/article/pii/S1746809422004694) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**A review of multimodal emotion recognition from datasets, preprocessing, features, and fusion methods**](https://www.sciencedirect.com/science/article/pii/S092523122300989X) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**A Sentiment and Emotion aware Multimodal Multiparty Humor Recognition in Multilingual Conversational Setting**](https://aclanthology.org/2022.coling-1.587/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**A Transformer-based Model with Self-distillation for Multimodal Emotion Recognition in Conversations**](https://ieeexplore.ieee.org/abstract/document/10109845/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**A Unified Transformer-based Network for Multimodal Emotion Recognition**](https://arxiv.org/abs/2308.14160) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**ACCOMMODATING MISSING MODALITIES IN TIME-CONTINUOUS MULTIMODAL EMOTION RECOGNITION**](https://ieeexplore.ieee.org/abstract/document/10388079/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations**](https://arxiv.org/abs/1810.02508) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Adaptive Multimodal Analysis for Speaker Emotion Recognition in Group Conversations**](https://arxiv.org/abs/2401.15164) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations**](https://ieeexplore.ieee.org/document/10109845) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**COntextualized GNN based Multimodal Emotion recognitioN**](https://aclanthology.org/2022.naacl-main.306/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Context- and Knowledge-Aware Graph Convolutional Network for Multimodal Emotion Recognition**](Context-_and_Knowledge-Aware_Graph_Convolutional_Network_for_Multimodal_Emotion_Recognition) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Convolutional Attention Networks for Multimodal Emotion Recognition from Speech and Text Data**](https://aclanthology.org/W18-3304/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Incongruity-Aware Dynamic Hierarchical Fusion for Multimodal Affect Recognition**](https://www.researchgate.net/profile/Yuanchao-Li-5/publication/370981780_Cross-Attention_is_Not_Enough_Incongruity-Aware_Multimodal_Sentiment_Analysis_and_Emotion_Recognition/links/64a48b888de7ed28ba74a8aa/Cross-Attention-is-Not-Enough-Incongruity-Aware-Multimodal-Sentiment-Analysis-and-Emotion-Recognition.pdf) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Cross-Language Speech Emotion Recognition Using Multimodal Dual Attention Transformers**](https://arxiv.org/abs/2306.13804) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Cross-modal credibility modelling for EEG-based multimodal emotion recognition**](https://iopscience.iop.org/article/10.1088/1741-2552/ad3987/meta) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Cross-modal fine-grained alignment and fusion network for multimodal aspect-based sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S0306457323002455) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Decoupled Multimodal Distilling for Emotion Recognition**](http://openaccess.thecvf.com/content/CVPR2023/html/Li_Decoupled_Multimodal_Distilling_for_Emotion_Recognition_CVPR_2023_paper.html) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Deep Emotional Arousal Network for Multimodal Sentiment Analysis and Emotion Recognition**](https://www.sciencedirect.com/science/article/pii/S1566253522000653) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Deep learning based multimodal emotion recognition using model-level fusion of audio–visual modalities**](https://www.sciencedirect.com/science/article/pii/S0950705122002593) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Deep Residual Adaptive Neural Network Based Feature Extraction for Cognitive Computing with Multimodal Sentiment Sensing and Emotion Recognition Process**](https://search.proquest.com/openview/5b82ac981d0318b3a57fe3bd2e737a49/1?pq-origsite=gscholar&cbl=52057) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**A Multimodal Transformer for Identifying Emotions and Intents in Social Conversations**](https://ieeexplore.ieee.org/abstract/document/9961847/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Emotion recognition from unimodal to multimodal analysis: A review**](https://www.sciencedirect.com/science/article/pii/S156625352300163X) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Emotion Recognition with Pre-Trained Transformers Using Multimodal Signals**](https://ieeexplore.ieee.org/abstract/document/9953852/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Emotion Recognition With Multimodal Transformer Fusion Framework Based on Acoustic and Lexical Information**](https://ieeexplore.ieee.org/document/9740502) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**EXPLOITING MODALITY-INVARIANT FEATURE FOR ROBUST MULTIMODAL EMOTION RECOGNITION WITH MISSING MODALITIES**](https://ieeexplore.ieee.org/abstract/document/10095836/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Exploiting Modality-Invariant Feature for Robust Multimodal Emotion Recognition with Missing Modalities**](Exploiting_Modality-Invariant_Feature_for_Robust_Multimodal_Emotion_Recognition_with_Missing_Modalities) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**A novel multimodal emotion recognition approach integrating face, body and text**](https://arxiv.org/abs/2211.15425) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt**](https://arxiv.org/abs/2305.10169) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**FUSING MODALITY-SPECIFIC REPRESENTATIONS AND DECISIONS FOR MULTIMODAL EMOTION RECOGNITION**](https://ieeexplore.ieee.org/abstract/document/10447035/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Learning Deep Representations for Multimodal Emotion Recognition**](https://dl.acm.org/doi/abs/10.1145/3581783.3612074) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**A graph network based multimodal fusion technique for emotion recognition in conversation**](https://www.sciencedirect.com/science/article/pii/S0925231223005507) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Learning Alignment for Multimodal Emotion Recognition from Speech**](https://arxiv.org/pdf/1909.05645) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Joint Modality Fusion and Graph Contrastive Learning for Multimodal Emotion Recognition**](https://arxiv.org/abs/2311.11009) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Learning discriminative multi-relation representations for multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S0020025523007107) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Leveraging Label Information for Multimodal Emotion Recognition**](https://arxiv.org/abs/2309.02106) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Multimodal Adversarial Learning Network for Conversational Emotion Recognition**](https://ieeexplore.ieee.org/abstract/document/10121331/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**A Multimodal Multi-Label Emotion, Intensity and Sentiment Dialogue Dataset for Emotion Recognition and Sentiment Analysis in Conversations**](https://aclanthology.org/2020.coling-main.393/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition**](https://arxiv.org/abs/2404.17113) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**RULE-BASED NETWORK FOR MULTIMODAL EMOTION RECOGNITION**](https://ieeexplore.ieee.org/abstract/document/10447930/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multi-Channel Weight-Sharing Autoencoder Based on Cascade Multi-Head Attention for Multimodal Emotion Recognition**](https://ieeexplore.ieee.org/abstract/document/9693238/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**An Attention-Based Correlation-Aware Multimodal Fusion Framework for Emotion Recognition in Conversations**](https://aclanthology.org/2023.acl-long.824/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Multilevel Transformer for Multimodal Emotion Recognition**](https://ieeexplore.ieee.org/abstract/document/10446812/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multimodal and Multi-view Models for Emotion Recognition**](https://arxiv.org/abs/1906.10198) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Multimodal Phased Transformer for Sentiment Analysis**](https://aclanthology.org/2021.emnlp-main.189/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Multimodal Prompt Transformer with Hybrid Contrastive Learning for Emotion Recognition in Conversation**](https://dl.acm.org/doi/abs/10.1145/3581783.3611805) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**Multimodal rough set transformer for sentiment analysis and emotion recognition**](https://ieeexplore.ieee.org/abstract/document/10263177/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multimodal Speech Emotion Recognition using Cross Attnention with Aligned Audio and Text**](https://ieeexplore.ieee.org/abstract/document/9414654/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**COntextualized GNN based Multimodal Emotion recognitioN**](https://aclanthology.org/2022.naacl-main.306/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**A Quantum-Inspired Adaptive-Priority-Learning Model for Multimodal Emotion Recognition**](https://aclanthology.org/2023.findings-acl.772/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**A REINFORCEMENT LEARNING FRAMEWORK FOR MULTIMODAL EMOTION RECOGNITION**](https://ieeexplore.ieee.org/abstract/document/10446459/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Self-adaptive Context and Modal-interaction Modeling For Multimodal Emotion Recognition**](https://aclanthology.org/2023.findings-acl.390/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Semi-Supervised Multimodal Emotion Recognition with Class-Balanced Pseudo-Labeling**](https://dl.acm.org/doi/abs/10.1145/3581783.3612864) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**SPEAKER-CENTRIC MULTIMODAL FUSION NETWORKS FOR EMOTION RECOGNITION IN CONVERSATIONS**](https://ieeexplore.ieee.org/abstract/document/10447720/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Token-channel compounded Cross Attention for Multimodal Emotion Recognition**](https://arxiv.org/abs/2306.13592) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Transformer-Based Deep-Scale Fusion Network for Multimodal Emotion Recognition**](https://ieeexplore.ieee.org/abstract/document/10254334/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Topic and Style-aware Transformer for Multimodal Emotion Recognition**](https://aclanthology.org/2023.findings-acl.130/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Towards Unified Multimodal Sentiment Analysis and Emotion Recognition**](https://arxiv.org/abs/2211.11256) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**USING AUXILIARY TASKS IN MULTIMODAL FUSION OF WAV2VEC 2.0 AND BERT FOR MULTIMODAL EMOTION RECOGNITION**](https://ieeexplore.ieee.org/abstract/document/10096586/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation**](https://arxiv.org/abs/2107.06779) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()


### Multimodal Aspect-based Sentiment Analysis (MABSA)

+ [**A Survey on Multimodal Aspect-Based Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/10401113/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**AoM Detecting Aspect-oriented Information for Multimodal**](https://arxiv.org/abs/2306.01004) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Cross-modal fine-grained alignment and fusion network for multimodal aspect-based sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S0306457323002455) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Cross-Modal Multitask Transformer for End-to-End Multimodal Aspect-Based Sentiment Analysis**](https://www.sciencedirect.com/science/article/pii/S0306457322001479) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Dual-Encoder Transformers with Cross-modal Alignment for Multimodal Aspect-based Sentiment Analysis**](https://aclanthology.org/2022.aacl-main.32/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()

+ [**Entity-Sensitive Attention and Fusion Network for Entity-Level Multimodal Sentiment Classification**](https://ieeexplore.ieee.org/document/8926404) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt**](https://arxiv.org/abs/2305.10169) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Few-shot joint multimodal aspect-sentiment analysis based on generative**](https://arxiv.org/abs/2305.10169) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Fusion with GCN and SE-ResNeXt Network for Aspect Based Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/document/10082618) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Hierarchical Interactive Multimodal Transformer for Aspect-Based Multimodal Sentiment Analysis**](https://ieeexplore.ieee.org/document/9765342) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**M2DF- Multi-grained Multi-curriculum Denoising Framework for Multimodal Aspect-based Sentiment Analysis**](https://arxiv.org/abs/2310.14605) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**MASAL A large-scale dataset for multimodal aspect-based sentiment**](https://www.sciencedirect.com/science/article/pii/S0925231221007931) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**MMCPR: A Chinese Product Review Dataset for Multimodal Aspect-Based Sentiment Analysis**](https://dl.acm.org/doi/abs/10.1007/978-3-031-23585-6_8) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**MOCOLNet A Momentum Contrastive Learning Network for Multimodal Aspect-Level Sentiment Analysis**](MOCOLNet_A_Momentum_Contrastive_Learning_Network_for_Multimodal_Aspect-Level_Sentiment_Analysis) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**ModalNet an Aspect-level sentiment classification model by exploring multimodal data with fusion discriminant attentional networks**](https://link.springer.com/article/10.1007/s11280-021-00955-7) [![Paper](https://img.shields.io/badge/Springer25-20B2AA)]()

+ [**MSRA- A Multi-Aspect Semantic Relevance Approach for E-Commerce via Multimodal Pre-Training**](https://dl.acm.org/doi/abs/10.1145/3583780.3615224) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**Multi-grained fusion network with self-distillation for aspect-based multimodal sentiment analysis**](https://www.sciencedirect.com/science/article/pii/S0950705124003599) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Multi-Interactive Memory Network for Aspect Based Multimodal Sentiment Analysis**](https://ojs.aaai.org/index.php/AAAI/article/view/3807) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Multi Interactive Memory Network for Aspect Based Multimodal Sentiment Analysis**](https://ojs.aaai.org/index.php/AAAI/article/view/3807) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Retrieving Users’ Opinions on Social Media with Multimodal Aspect-Based Sentiment Analysis**](https://ieeexplore.ieee.org/abstract/document/10066699/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Targeted Multimodal Sentiment Classification Based on**](https://www.ijcai.org/proceedings/2022/0622.pdf) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Targeted Aspect-Based Multimodal Sentiment Analysis An Attention Capsule Extraction and Multi-Head Fusion Network**](https://ieeexplore.ieee.org/document/9606882) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis 2**](https://arxiv.org/abs/2204.07955) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis**](https://arxiv.org/abs/2204.07955) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**VistaNet Visual Aspect Attention Network for Multimodal Sentiment Analysis**](https://ojs.aaai.org/index.php/AAAI/article/view/3799) [![Paper](https://img.shields.io/badge/Other25-A9A9A9)]()

+ [**Visual Attention Model for Name Tagging in Multimodal Social Media**](https://aclanthology.org/P18-1185/) [![Paper](https://img.shields.io/badge/ACL25-DA70D6)]()


### Multimodal Multi-label Emotion Recognition (MMER)

+ [**Label Distribution Adaptation for Multimodal Emotion Recognition with Multi-label Learning**](https://dl.acm.org/doi/abs/10.1145/3607865.3613183) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

+ [**Multi-Label Multimodal Emotion Recognition With Transformer-Based Fusion and Emotion-Level Representation Learning**](https://ieeexplore.ieee.org/abstract/document/10042438/) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()

+ [**Multimodal Emotion Classification**](https://dl.acm.org/doi/abs/10.1145/3308560.3316549) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()


### Multimodal Emotion-cause Pair Extraction (MECPE)

+ [**LastResort at SemEval-2024 Task 3: Exploring Multimodal Emotion Cause Pair Extraction as Sequence Labelling Task**](https://arxiv.org/abs/2404.02088) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()

+ [**Multimodal Emotion-Cause Pair Extraction in Conversations**](https://ieeexplore.ieee.org/document/9969873) [![Paper](https://img.shields.io/badge/IEEE25-1E90FF)]()


### Multimodal Affective Computing with Missing_modality

+ [**Multimodal Reconstruct and Align Net for Missing Modality Problem in Sentiment Analysis**](https://link.springer.com/chapter/10.1007/978-3-031-27818-1_34) [![Paper](https://img.shields.io/badge/Springer25-20B2AA)]()

+ [**integrating consistency and difference networks by transformer for multimodal sentiment analysis**](https://link.springer.com/article/10.1007/s10489-023-04869-x) [![Paper](https://img.shields.io/badge/Springer25-20B2AA)]()

+ [**assisted Multimodal Sentiment Analysis under Uncertain Missing Modalities**](https://dl.acm.org/doi/abs/10.1145/3477495.3532064) [![Paper](https://img.shields.io/badge/ACM25-FFA500)]()

