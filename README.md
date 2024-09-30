# Dataengine Tasks - Renswick Delvar

## Folder Structure
    data/                                   # Contains dataset files
        Machine Learning Challenge.ods      # Primary dataset in ODS format
        Racing Data Set.zip                 # Task 3 dataset (zipped)

    Task2/                                  # Task 2 Jupyter Notebook files
        Task2-Notebook.ipynb                # Task 2 notebook for processing and analysis

    Task3/                                  # Task 3 scripts and related files
        main.py                             # Main Python script for Task 3 execution
        prompt.py                           # Supporting script for prompt management

    README.md                               # Project documentation (this file)
    requirements.txt                        # List of dependencies for the project


## Data Science Task: Customer Segmentation and Anomaly Detection
This project performs anomaly detection followed by customer segmentation using various machine learning techniques. The goal is to identify outliers in the customer data and segment customers into distinct groups based on their behavior, demographics, and engagement metrics.

### Pesudo Code for Data Science Task
START

### Step 1: Load the dataset
 - LOAD 'customer_data.ods'

### Step 2: Initial Data Preprocessing
 - DROP 'consumer_id' column (not useful for analysis)
 - DROP 'account_status' column (constant values)
 - REMOVE duplicates from the dataset

### Step 3: Handle Missing Values
* IF missing values in 'gender' or 'customer_age':
    APPLY MICE imputation on 'gender' and 'customer_age'
    REPLACE missing values

### Step 4: Feature Engineering
#### Create new features based on customer behavior
 - CREATE new column 'ctr' = 'total_offer_clicks' / 'total_offer_impressions'
 - CREATE new column 'redemption_rate' = 'total_offers_redeemed' / 'total_offer_clicks'
 - CREATE new column 'unique_offer_ctr' = 'unique_offer_clicked' / 'unique_offer_impressions'

#### Log transformation of skewed features
 - FOR each feature in ['customer_age', 'account_age', 'total_offer_clicks', 'account_last_updated', 
                     'total_offer_impressions', 'total_offers_redeemed', 'unique_offer_clicked', 
                     'unique_offer_impressions']:
     - CONVERT feature to numeric (to handle errors)
     - APPLY log transformation: 'log_feature' = log('feature' + 1)

#### Convert 'app_downloads' to binary format (1 if app is downloaded, else 0)
 - TRANSFORM 'app_downloads' to 1 if value == 1, else 0

### Step 5: Handle Extreme Distributions with K-Modes Clustering
#### Clustering based on categorical columns (demographic information)
 - SELECT columns starting with 'has_'
 - APPLY K-Modes clustering on selected 'has_' columns
 - ADD new column 'demographic_cluster' with cluster labels

#### Clustering based on redemption-related features
 - SELECT columns with 'redemptions' in the name
 - APPLY K-Modes clustering on 'redemption' columns
 - ADD new column 'redemption_cluster' with cluster labels

### Step 6: Normalize Data (for both numerical and categorical features)
 - FOR each numerical column:
     - PPLY Min-Max normalization (scale values between 0 and 1)

### Step 7: Anomaly Detection (using multiple techniques)
 - INITIALIZE anomaly detection models (LOF, Isolation Forest, DBSCAN, One-Class SVM, Autoencoders)
 - FOR each anomaly detection model:
     - DETECT anomalies and store their indices

#### Combine anomalies from all models and treat them as outliers
 - FILTER outliers from the dataset

### Step 8: Customer Segmentation (using clustering models)
#### K-Prototypes Clustering for mixed data types
 - APPLY K-Prototypes to cluster customers into groups

#### Gaussian Mixture Model (GMM) for probabilistic clustering
 - APPLY GMM for clustering

#### Hierarchical Clustering for hierarchical segmentation
 - APPLY Hierarchical Clustering and visualize using Dendrogram

#### Determine Optimal Number of Clusters
 - USE Elbow method for K-Prototypes
 - USE BIC for GMM
 - USE Dendrogram for Hierarchical Clustering

### Step 9: Visualize Clusters with PCA (2 components)
 - PERFORM PCA to reduce data to 2 components
 - PLOT customer groups on a 2D plane based on the first two principal components


END

## Gen AI Chatbot
This Generative AI chatbot is designed to provide intelligent responses by leveraging the `Google Gemini Pro` model for text embeddings `(models/embedding-001)` and using `ChromaDB` as the vector database for efficient similarity searches. The application allows users to upload PDF and text files, which are then processed, chunked, and indexed into ChromaDB. Users can interact with the chatbot by asking questions, which triggers a search for the most relevant document chunks, followed by a response generated using a conversational chain built on `LangChain`. The bot is hosted via `Streamlit`, offering an interactive and user-friendly interface.

### Configuration & Setup

Follow these steps to configure and set up the Generative AI Chatbot:

1. Clone the Repository to local, using the below command
 `git clone -b main https://github.com/renswickd/Dataengine-Tasks.git`

2. Set Up Environment Variables
Create a .env file in the root directory with the following content:

`GOOGLE_API_KEY=your-google-api-key`

This key is required for accessing Google Gemini Pro Model via LangChain.

3. Install Dependencies
Ensure you have Python 3.x installed. Then, install the required libraries by running:

`pip install -r requirements.txt`

Dependencies include Streamlit, LangChain, Google Gemini Pro Embeddings, ChromaDB, PyPDF2, and other essential libraries.

4. Configure ChromaDB
ChromaDB will be used as the vector store to handle text embeddings. By default, the vector store is saved in the chroma_db_directory. No additional setup is required, but ensure that this directory is accessible.

5. Run the Application
You can start the Streamlit app by running the following command:

`streamlit run main.py`

This will launch the web application, where you can navigate between the ETL Pipeline (for developers) and the Chatbot (for users).

### Pseudo code
START

#### Step 1: Set up Environment
    LOAD environment variables (API keys) from .env file
    IF API key is missing, DISPLAY error message, else configure API

#### Step 2: Define Helper Functions
    DEFINE function 'get_pdf_text':
        INPUT: list of PDF or text files
        OUTPUT: combined text from all files
        DESCRIPTION: Extract text from PDF and text files

    DEFINE function 'get_text_chunks':
        INPUT: text data, chunk size, chunk overlap
        OUTPUT: list of text chunks
        DESCRIPTION: Split the text into smaller chunks for embedding

    DEFINE function 'get_vector_store':
        INPUT: text chunks, collection name, file name
        OUTPUT: vector store with embedded chunks
        DESCRIPTION: Generate embeddings and store them in Chroma DB with metadata
    
    DEFINE function 'user_input':
        INPUT: user question, collection name
        PROCESS: 
            - Fetch the vector store for the text chunks
            - Perform similarity search based on user question
            - Retrieve the most relevant documents from Chroma DB
            - Generate a conversational response using the LLM
        OUTPUT: Display response and retrieved documents for demo

#### Step 3: Main Function
    DISPLAY sidebar for page navigation with options "ETL Pipeline" and "Chatbot"

    IF page is "ETL Pipeline":
        Step 3.1: Upload Documents
            PROMPT user to upload PDF or text files
            IF user clicks 'Extract Data', EXTRACT text using 'get_pdf_text'
        
        Step 3.2: Chunk Text and Indexing
            PROMPT user to input chunk size and chunk overlap values
            IF user clicks 'Chunking & Indexing', CHUNK the extracted text using 'get_text_chunks'
        
        Step 3.3: Generate Embeddings and Store in Vector Database
            PROMPT user for collection name
            IF user clicks 'Generate Embeddings & Store Vector Store', 
                - CREATE embeddings for text chunks using 'get_vector_store'
                - SAVE the embeddings in Chroma DB

    IF page is "Chatbot":
        Step 3.4: Chatbot Interaction
            PROMPT user to enter a question
            IF user submits a question, 
                - RETRIEVE the most relevant documents from Chroma DB
                - GENERATE a conversational response using 'user_input'
                - DISPLAY response and retrieved documents (for demo purposes)

END

### Addition Enhancements to Make the Application More Robust - (If time permits)

1. Add Authentication and Access Control: Secure the "ETL Pipeline" with Streamlit authentication (e.g., OAuth, Firebase) to ensure only authorized users can upload documents.
2. Optimize Document Chunking with Adaptive Chunking: Use NLP tools like nltk or spaCy for sentence boundary-based chunking, ensuring coherent chunks instead of fixed sizes.
3. LangChain Agents for Advanced Tools: Integrate LangChain agents to automate document summarization, question answering, and real-time decision-making using pre-trained LLMs and task-specific tools.
4. Add Feedback Loop for Continuous Learning: Implement a feedback mechanism to adjust similarity weights based on user ratings, refining vector retrieval and responses.
5. Auto-Suggest for Common Questions: Use keyword extraction (e.g., TF-IDF, BERT) to implement auto-suggestion of questions based on document content as the user types.