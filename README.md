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

+ [**Compile Scene Graphs with Reinforcement Learning**](https://arxiv.org/pdf/2504.13617) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]()  [![Star](https://img.shields.io/github/stars/gpt4vision/R1-SGG.svg?style=social&label=Star)](https://github.com/gpt4vision/R1-SGG) 

+ [**PRISM-0: A Predicate-Rich Scene Graph Generation Framework for Zero-Shot Open-Vocabulary Tasks**](https://arxiv.org/pdf/2504.00844) [![Paper](https://img.shields.io/badge/arXiv25-b22222)]() 
