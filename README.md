# Privacy-concerns-in-machine-learning-models-trained-on-medical-data

# problem: 
- Machine learning models trained on medical data can reveal sensitive information about individuals, which can be exploited by attackers. Ensuring the privacy of such data is a major concern in machine learning applications.

# Background:
- Machine learning models trained on medical data have the potential to revolutionize healthcare by providing personalized diagnoses and treatments. However, these models can also reveal sensitive information about individuals, such as their medical history or genetic information. This information can be exploited by attackers, who can use it for identity theft or other malicious purposes. As a result, ensuring the privacy of medical data is a major concern in machine learning applications.

# Impact:
- Privacy breaches in machine learning models trained on medical data can have serious consequences for individuals, including identity theft and discrimination. For example, an attacker could use a machine learning model trained on medical data to identify individuals with certain medical conditions, and then use that information to discriminate against them in employment or insurance. 

## Objective

To **develop and compare multiple machine learning models** on medical data, evaluate the privacy risks, and implement techniques like:

- **Federated Learning**
- **Split Learning**
- **Differential Privacy (DP)**

  the objective decided within the project was to **predict ICU mortality** using the dataset . 

Our goal is to achieve high model performance **without compromising patient privacy**, ensuring compliance with data protection regulations (HIPAA, GDPR).

## Dataset

###  Source
We used the **MIMIC-III (v1.4)** dataset, a sample dataset available with critical care database developed by the MIT Lab for Computational Physiology.

- 📎 Link: https://physionet.org/content/mimiciii/1.4/

  ## 🔬 Methods Implemented


| Baseline MLP | Basic neural network without privacy 
| Federated Learning (FL) | Distributed learning across clients 
| Split Learning (SL) | Model split across client-server 
| Differential Privacy (DP) | Adds noise to gradients / outputs 

