<div align="center">

# Uni-SciSum: Enhancing Large Language Models for Scientific Multimodal Summarization with Multimodal Output

![Paper](https://img.shields.io/badge/paper--blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

</div>


## ❓ What is Uni-SciSum

Uni-SciSum is a novel multimodal scientific summarisation model for multimodal output. VAT-Sum aims to enable LMs to effectively utilize textual, visual and audio content for scientific document summarisation. With the use of Q-Former, our model can effectively fuse these multimodal inputs and feed them into the lightweight LM for efficient summarisation. This lightweight LM facilitates efficient downstream adaptation without sacrificing performance.

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
   
   You need to resolve around 10G of space for the LMs and data, you can run the following code for the reproduction of the model performance:
   ```shell
   bash run_finetune.sh
   ```

## 🔎 Citation

```
[]
```


## 📬 Contact

If you have any inquiries, suggestions, or wish to contact us for any reason, we warmly invite you to email us at allentan@ln.hk.
