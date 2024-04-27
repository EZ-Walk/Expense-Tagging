# Expense Dashboard

Status: To Do
ID: PRO-42
Last edited time: April 27, 2024 3:34 PM
Work Status: Bidding
✅ Tasks: Expense data processing (https://www.notion.so/Expense-data-processing-fccafff37a5a4062807efd88ff0c53f4?pvs=21), Expense tagging feature engineering (https://www.notion.so/Expense-tagging-feature-engineering-ed560486032548eba58dad28dd7fca60?pvs=21), Expense tagging model (https://www.notion.so/Expense-tagging-model-9e8d27cb87084b818c71096496831690?pvs=21)

# Mission: Make my expenses more consumable

# MVP 1: Expense tagging

A streamlit app that tags a csv of financial expenses with preset tags and lets you redownload it

Tags: Public transit, Gas, Vehicle maintenance, Dining out, Recreation, Groceries, Rent, Utilities, Insurance, Subscription, Health,

# Problem: unmanaged finances

My small credit union does nothing to breakdown my spending into a consumable format. They do however, provide a download that supplies the date, vendor description and amount of the purchase. 

# Approach/Design

1. **Data Preprocessing:**
    - Extract relevant information from the vendor descriptions (e.g., “MASABI_RTD 1600 Blake St 303-299-6000 COUS”).
    - Parse the dates and convert them to a consistent format (e.g., MM/DD/YYYY).
    - Extract the numerical amount (in USD) from each expense.

1. **Feature Engineering:**
    - Create features from the processed data:
        - **Text Features (Vendor Descriptions):**
            - Use techniques like bag-of-words or TF-IDF to represent the vendor descriptions as numerical vectors.
            - [Consider using pre-trained language models (e.g., BERT, GPT) to generate embeddings for the descriptions1](https://www.width.ai/post/document-classification).
        - **Date Features:**
            - Convert dates to numerical features (e.g., day of the week, month, etc.).
        - **Amount Features:**
            - Normalize the amounts to a common scale (e.g., z-score normalization).

1. **Training a Model:**
    - Collect a labeled dataset where each expense is associated with one or more tags (e.g., “dining,” “groceries,” etc.).
    - Train a multi-label classification model (e.g., logistic regression, neural network, or transformer-based model) using the features.
    - The model should predict the relevant tags for each expense.
2. **Evaluation and Fine-Tuning:**
    - Split your dataset into training and validation sets.
    - Evaluate the model’s performance using metrics like precision, recall, F1-score, and accuracy.
    - Fine-tune the model based on the validation results.
3. **Inference:**
    - Apply the trained model to new expenses:
        - Extract features from the new expense (vendor description, date, and amount).
        - Use the model to predict the relevant tags.
4. **Tag Assignment:**
    - Based on the model’s predictions, assign the appropriate tags to each expense.
5. **GPT-3-Based Pipeline (Optional):**
    - [If you have access to GPT-3, you can use it for few-shot learning and fine-tuning to improve classification accuracy1](https://www.width.ai/post/document-classification).
    - Provide examples of labeled expenses to GPT-3 and let it learn the context for better predictions.

Remember that the quality of your model depends on the quality and size of your labeled dataset. The more diverse and representative your data, the better your model will perform. [Good luck with your expense classification task! 📊💡1](https://www.width.ai/post/document-classification)[2](https://towardsdatascience.com/categorising-short-text-descriptions-machine-learning-or-not-d3ec8de8c40)[3](https://stackoverflow.com/questions/57001614/tensorflow-implementation-for-bank-transaction-classification)