# Aspects_based_financial_sentiment_analysis

# Overview : 
The IT457 project focuses on Aspect-based Financial Sentiment Analysis. In this project, we analyze sentiment scores of sentences from the FiQA Task 1 dataset. Our approach involves examining numerical values within the sentences, as these numbers significantly influence sentiment scores. For instance, if one sentence includes "3%" and another contains "30%", even though they express similar sentiments, the second sentence will have a higher sentiment score due to the larger percentage.

In addition to numerical values, we also account for root words—such as "high," "down," and "up"—that impact sentiment scores. Therefore, the overall sentiment score is determined by both the numerical context and the presence of these key aspect-related terms.

# Methodology :

The methodology of the project as shown in figure and implemented in (IT457_Final_project):

![WhatsApp Image 2024-09-05 at 1 02 55 AM (1)](https://github.com/user-attachments/assets/7c5db4b0-2eb8-45c2-b7b9-1c5af07fbed6)

1) Input Module (FiQA dataset Headlines): Provides financial news headlines from the FiQA dataset as input.

2) Language Model (BERT): Extracts contextual embeddings from headlines using the BERT language model.

3) Numerical Encoder: Processes numerical values in headlines, capturing their relevance through dependency trees and a Digit CNN with max-pooling
   
4) Dictionary Creation: Creates a set of auxiliary terms to identify and embed relevant financial aspects.
   
6) Aspect Embedding: Generates embeddings for specific financial aspects based on auxiliary terms in a custom dictionary.

7) Multi-head Attention Encoder: Applies attention mechanisms to focus on multiple financial aspects within the embeddings.

8) Sentiment Score: To calculate sentiment scores, we combine BERT embeddings, numerical values, and attention mechanisms, generating info_sentiment_scores. We split the FiQA dataset (498 samples) into 85% for training and 15% for testing, training a deep neural network on the training set. Finally, we test with the remaining samples and compare predicted sentiment scores with the actual scores for evaluation.

9) Evaluation Metrices : MSE and MAE

10) Creating model as sentiment_model.pkl in which we can predict any sentiment scores of any sentences along with its aspects(tested in IT457_final_result.ipynb).

# User Interface Design :

In input.csv we took 12 samples which consists of sentences and its aspects for testing this model (sentiment_model.pkl) and this done with user interface design used as : html,CSS,flask. 

## Input :
The given figure main page we took input from input.csv file any row and after predict button it will go to result page:

<img width="833" alt="userinterfacedesign" src="https://github.com/user-attachments/assets/2806c16b-2c04-4c4e-be75-a390ad3c98c7">

## The 2 figures shows the output of particular sentences which we passed in input : 

## Output1 : 

<img width="874" alt="userinterfacedesign1" src="https://github.com/user-attachments/assets/afcb502b-1d89-48ab-9652-7897d7a5e799">

## Output2 : 

<img width="829" alt="userinterfacedesign2" src="https://github.com/user-attachments/assets/c8b026e1-4bec-43b2-a65f-b31c0c910e11">




The sentiment_model.pkl file is available at the following link : " https://drive.google.com/file/d/1ZQKp07LIX7EmeGoAk8SxpiNP2aSxOWQY/view?usp=sharing " Simply visit this link to download the model.
