# GTU Regulations Q&A System
![image](https://github.com/user-attachments/assets/c565992c-b121-4b63-b9bc-8cec1bf1a461)
![image](https://github.com/user-attachments/assets/7261e7a3-96cf-411c-81f9-92c093320397)



## Prerequisits
- Please first download the model from the followin link and add it to this folder
https://drive.google.com/drive/folders/1Jj_PlBxsGhAz6fxTEorAjY-s-UQ7vZzq?usp=sharing
(I am very sorry about this inconvenience, the models file didnt fit)

## Features
- Simple web interface for asking questions
- Section-based context navigation
- Real-time answers with confidence scores
- Pre-trained on university regulations

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the displayed URL

## How to Use
1. Select a regulation section from the dropdown menu
3. Type your question in Turkish
4. Click "Get Answer" to receive a response

## About the Project
Data Collection
To collect the data, the PDF document of the GTU Rules and Regulations file was extracted using a custom Python script and the PyPDF library, converted into JSON format for accessibility when annotating the data.

Model Selection
The model chosen to be finetuned to the GTU Rules and Regulations dataset was the Turkish SQuAD Model: Question Answering [1], pre-finetuned with the Turkish language version of SQuAD, TQuAD.

Fine-Tuning the Model
The Turkish QA model was finetuned using the ‘Transformers’ library. 
Finetuning metrics

Metric	Value
Number of Epochs	4
Learning Rate	0.0002
Weight Decay	0.01

Model Evaluation
The model was evaluated using F1 Score and Exact Match metrics.
![image](https://github.com/user-attachments/assets/c1450bc3-e2a5-4afa-8c35-f3bd974877da)


