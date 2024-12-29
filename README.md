<div align="center">

# Uni-SciSum: Enhancing Large Language Models for Scientific Multimodal Summarization with Multimodal Output

![Paper](https://img.shields.io/badge/paper--blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

</div>


## ❓ What is Uni-SciSum

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
[]
```


## 📬 Contact

If you have any inquiries, suggestions, or wish to contact us for any reason, we warmly invite you to email us at allentan@ln.hk.
