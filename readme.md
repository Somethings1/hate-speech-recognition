# Hate speech recognition
---

# 1. Executive Summary

This document outlines a state-of-the-art, multi-phase framework for developing a high-accuracy hate speech detection model specifically for the Vietnamese language. The problem is uniquely challenging due to linguistic nuances, including slang, phonetic substitutions, coded language, and context-dependency. Our proposed solution moves beyond traditional methods by employing a hybrid approach that leverages Large Language Models (LLMs) for scalable data labeling, a custom-trained Transformer-based model for efficient real-time inference, and a continuous learning loop for long-term adaptability. This framework is designed to be robust against adversarial user behavior and is feasible to implement using freely available platforms like Google Colab or Kaggle.

# 2. Phase 1: Scalable Dataset Creation via Human-in-the-Loop Weak Supervision

The primary bottleneck in building a specialized classifier is the lack of a large, high-quality labeled dataset. Manual labeling is slow, expensive, and inconsistent. We will overcome this using a semi-supervised approach.

## 2.1. LLM-Powered Pre-Labeling:

We will utilize a state-of-the-art LLM (e.g., Gemini, GPT-4) as a sophisticated "labeling assistant." By crafting a detailed prompt with a clear definition of hate speech, counter-examples, and few-shot examples of nuanced cases, we will instruct the LLM to process batches of unlabeled comments. The LLM will return structured JSON data, including the comment, a hate_score, a confidence_score, and a justification for its decision.

## 2.2. Human-in-the-Loop (HITL) Verification:

Human expertise is our most valuable resource. Instead of reviewing every comment, reviewers will focus only on comments where the LLM's confidence_score is below a set threshold (e.g., < 0.85). This focuses human effort on the most ambiguous and difficult cases. This process will create a high-quality "Golden Dataset" for model training.

Supporting Literature:


Concept: This methodology is a form of Weak Supervision, where higher-level, noisy signals (from the LLM) are used to create training data.

- Paper 1: Ratner, A., et al. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. This foundational paper introduces the paradigm of using programmatic and heuristic rules (of which an LLM is a highly advanced form) to create large-scale training datasets without manual labeling of every item.


- Paper 2: Gilardi, F., et al. (2023). ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks. This study provides empirical evidence that modern LLMs can outperform human annotators in consistency and quality for many text classification tasks, validating their use as a primary labeling tool.


# 3. Phase 2: Robust Model Development and Training

The core of our system will be a custom-trained model designed to be efficient, accurate, and resilient to adversarial tactics.

## 3.1. Core Architecture:

Vietnamese-Specific Transformer: We will use PhoBERT as our base model. PhoBERT is a Transformer model pre-trained on a massive (20GB) corpus of Vietnamese text, giving it a deep, innate understanding of the language's syntax and semantics.

- Paper 3: Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. This paper introduces the model and demonstrates its state-of-the-art performance on various Vietnamese NLP tasks.

## 3.2. Handling Character & Linguistic Obfuscation:

Users will intentionally use evasive language (e.g., p4rky for "Bắc Kỳ", mạy mè for "mẹ mày"). We will address this with a two-pronged strategy:

### 1. Data Augmentation:

We will programmatically create augmented training examples by applying random character swaps (e -> 3, a -> 4) and slang substitutions based on a curated dictionary.

### 2. Character-Aware Architecture:

We will enhance our model by incorporating a Character-level Convolutional Neural Network (CharCNN) before the Transformer layers. The CharCNN learns representations from character sequences, making the model resilient to typos and unseen obfuscations that would break a standard subword tokenizer.

- Paper 4: Kim, Y., et al. (2016). Character-Aware Neural Language Models. This paper demonstrates the effectiveness of using character-level inputs to handle morphological richness and out-of-vocabulary words, the same principles that apply to obfuscated text.

## 3.3. Incorporating External Context: Much of hate speech is context-dependent (e.g., "Lẩu Gà Bình Thuận" is only hateful in a video about LGBTQ+ individuals). The model must be given this context.

### 1. Input Formulation:

The model's input will not be the comment alone, but a concatenated string of video metadata and the comment, structured as: [CLS] <video_title> [SEP] <video_tags> [SEP] <comment_text> [SEP].

### 2. Mechanism:

The Transformer's self-attention mechanism is inherently designed to understand relationships between tokens. It will learn to correlate words in the comment with words in the title and tags to make a contextually-informed decision.

* Paper 5: Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. The original BERT paper explains the use of the [SEP] token to handle multiple text segments, providing the technical foundation for our context-aware input strategy.

# 4. Phase 3: Deployment and Iterative Improvement

A static model will quickly become outdated. Our framework includes a continuous learning loop.

## 4.1. Deployment:

The fine-tuned, specialized model (e.g., our enhanced PhoBERT) will be deployed for real-time inference. It is significantly faster and cheaper to run at scale than making an API call to a large LLM for every comment.

## 4.2. Feedback Loop (Active Learning):

The deployed model will log cases with low confidence scores or those flagged by users. These "hard cases" will be sent back to the Phase 1 pipeline for LLM-assisted labeling and human review.

## 4.3. Periodic Re-training:

The Golden Dataset will be continuously augmented with these new, challenging examples. The model will be periodically re-trained to adapt to evolving slang, new obfuscation techniques, and emerging hateful narratives.

# 5. Conclusion

This framework directly addresses the core challenges of Vietnamese hate speech detection by combining the scalable labeling power of LLMs with the efficiency and specificity of a custom-trained, context-aware Transformer model. By focusing on data quality, robustness to adversarial input, and a continuous feedback loop, we can build a system that is not only effective at launch but also adaptable to the dynamic nature of online communication.

