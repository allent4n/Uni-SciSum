<div align="center">

# VAT-Sum: A Lightweight Language Model for Vision-Audio-Text Scientific Multimodal Summarisation with Multimodal Output

![Paper](https://img.shields.io/badge/paper--blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

</div>


## ❓ What is VAT-Sum 

VAT-Sum is a novel multimodal scientific summarisation model for multimodal output. VAT-Sum aims to enable LMs to effectively utilize textual, visual and audio content for scientific document summarisation. With the use of Q-Former, our model can effectively fuse these multimodal inputs and feed them into the lightweight LM for efficient summarisation. This lightweight LM facilitates efficient downstream adaptation without sacrificing performance.

## ⚡️ Quickstart
1. **Clone the GitHub Repository:** 

   ```shell
   git clone https://github.com/allent4n/VAT-Sum.git
   ```

2. **Set Up Python Environment:** 

   ```shell
   cd VAT-Sum-main
   virtualenv -p python3.12 venv
   source venv/bin/activate
   ```

3. **Install SEA Dependencies:** 
   ```shell
   pip install -r requirements.txt
   ```
   
4. **Reproduce Results:**
   
   You need to resolve around 10G of space for the LMs and data, you can run the following code for the reproduction of textual summarisation:
   ```shell
   bash textual_sum.sh
   ```

   Run the following code for the reproduction of textual summarisation:
   ```shell
   bash graphical_sum.sh
   ```

## 🔎 Citation

```
[]
```


## 📬 Contact

If you have any inquiries, suggestions, or wish to contact us for any reason, we warmly invite you to email us at allentan@ln.hk.
