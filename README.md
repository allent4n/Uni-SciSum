<div align="center">

# Uni-SciSum: Enhancing Large Language Models for Scientific Multimodal Summarization with Multimodal Output

[![Paper](https://img.shields.io/badge/paper--blue)](https://aclanthology.org/2025.coling-industry.22/)
![Python](https://img.shields.io/badge/python-3.12-blue)

</div>


## ❓ What is Uni-SciSum （published in COLING-2025）

Uni-SciSum is a novel multimodal scientific summarisation model for multimodal output. Uni-SciSum aims to enable LLMs to effectively utilize textual, visual and auditoral content for scientific summarisation. Our model connects unimodal encoders to multimodal decoders via BridgeNet. During pretraining, the learnable queries in BridgeNet learn to extract modality-specific features from the encoders. During downstream tasks, the decoder generates embeddings based on different inputs and outputs (guided by the prompt and the learned queries), which the LLM then decodes into the target text summary and graphical abstract (GA).
<img width="1342" alt="framework_v3" src="https://github.com/user-attachments/assets/d51ab625-37eb-4abd-8ad2-6f2bb109177f" />


## ⚡️ Quickstart
1. **Clone the GitHub Repository:** 

   ```shell
   git clone https://github.com/allent4n/Uni-SciSum.git
   ```

2. **Set Up Python Environment:** 

   ```shell
   cd Uni-SciSum-main
   virtualenv -p python3.12 venv
   source venv/bin/activate
   ```

3. **Install SEA Dependencies:** 
   ```shell
   pip install -r requirements.txt
   ```
   
4. **Reproduce Results:**
   
   You need to resolve around 10G of space for the LLMs and data, you can run the following code for the reproduction of the model performance:
   ```shell
   bash run_finetune.sh
   ```

## 🔎 Citation

```
@inproceedings{tan-etal-2025-enhancing,
    title = "Enhancing Large Language Models for Scientific Multimodal Summarization with Multimodal Output",
    author = "Tan, Zusheng  and
      Zhong, Xinyi  and
      Ji, Jing-Yu  and
      Jiang, Wei  and
      Chiu, Billy",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven  and
      Darwish, Kareem  and
      Agarwal, Apoorv",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics: Industry Track",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-industry.22/",
    pages = "263--275",
    abstract = "The increasing integration of multimedia such as videos and graphical abstracts in scientific publications necessitates advanced summarization techniques. This paper introduces Uni-SciSum, a framework for Scientific Multimodal Summarization with Multimodal Output (SMSMO), addressing the challenges of fusing heterogeneous data sources (e.g., text, images, video, audio) and outputting multimodal summary within a unified architecture. Uni-SciSum leverages the power of large language models (LLMs) and extends its capability to cross-modal understanding through BridgeNet, a query-based transformer that fuses diverse modalities into a fixed-length embedding. A two-stage training process, involving modal-to-modal pre-training and cross-modal instruction tuning, aligns different modalities with summaries and optimizes for multimodal summary generation. Experiments on two new SMSMO datasets show Uni-SciSum outperforms uni- and multi-modality methods, advancing LLM applications in the increasingly multimodal realm of scientific communication."
}
```


## 📬 Contact

If you have any inquiries, suggestions, or wish to contact us for any reason, we warmly invite you to email us at allentan@ln.hk.
