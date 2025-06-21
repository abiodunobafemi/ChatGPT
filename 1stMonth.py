{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c0e5e2d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package omw-1.4 to\n",
      "[nltk_data]     /Users/abiodunobafemi/nltk_data...\n",
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     /Users/abiodunobafemi/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     /Users/abiodunobafemi/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     /Users/abiodunobafemi/nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Preprocessed data saved to: /Users/abiodunobafemi/Documents/Research/NCUR 2024/Data/clean_chatgpt_1stMonth.csv\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "import nltk\n",
    "import os\n",
    "nltk.download('omw-1.4')\n",
    "\n",
    "\n",
    "# Set NLTK data path to a directory without omw-1.4 resource\n",
    "nltk.data.path = [os.path.join(os.path.expanduser('~'), 'nltk_data')]\n",
    "\n",
    "# Load the data into a pandas DataFrame\n",
    "file_path = '/Users/abiodunobafemi/Documents/Research/NCUR 2024/Data/chatgpt_1stMonth.csv'\n",
    "data = pd.read_csv(file_path)\n",
    "\n",
    "# Download NLTK resources if not already downloaded\n",
    "nltk.download('stopwords')\n",
    "nltk.download('punkt')\n",
    "nltk.download('wordnet')\n",
    "\n",
    "# Function for preprocessing text\n",
    "def preprocess_text(text):\n",
    "    # Remove URLs\n",
    "    text = re.sub(r'http\\S+', '', text)\n",
    "    # Remove special characters and emojis\n",
    "    text = re.sub(r'[^\\w\\s]', '', text)\n",
    "    # Convert text to lowercase\n",
    "    text = text.lower()\n",
    "    return text\n",
    "\n",
    "# Function for tokenization and stopwords removal\n",
    "def tokenize_and_remove_stopwords(text):\n",
    "    # Tokenize text\n",
    "    tokens = word_tokenize(text)\n",
    "    # Remove stopwords\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    filtered_tokens = [word for word in tokens if word not in stop_words]\n",
    "    return filtered_tokens\n",
    "\n",
    "# Function for lemmatization\n",
    "def lemmatize_text(tokens):\n",
    "    lemmatizer = WordNetLemmatizer()\n",
    "    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]\n",
    "    return lemmatized_tokens\n",
    "\n",
    "# Apply preprocessing steps to the tweet text\n",
    "data['clean_text'] = data['tweet'].apply(preprocess_text)\n",
    "\n",
    "# Tokenize and remove stopwords from the cleaned text\n",
    "data['tokens'] = data['clean_text'].apply(tokenize_and_remove_stopwords)\n",
    "\n",
    "# Lemmatize the tokens\n",
    "data['lemmatized_tokens'] = data['tokens'].apply(lemmatize_text)\n",
    "\n",
    "# Save the preprocessed data to a new CSV file\n",
    "output_file_path = '/Users/abiodunobafemi/Documents/Research/NCUR 2024/Data/clean_chatgpt_1stMonth.csv'\n",
    "data.to_csv(output_file_path, index=False)\n",
    "\n",
    "# Display a message indicating the completion of the saving process\n",
    "print(\"Preprocessed data saved to:\", output_file_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d56dbbe6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.decomposition import LatentDirichletAllocation\n",
    "\n",
    "# Load the preprocessed data into a pandas DataFrame\n",
    "file_path = '/Users/abiodunobafemi/Documents/Research/NCUR 2024/Data/clean_chatgpt_1stMonth.csv'\n",
    "data = pd.read_csv(file_path)\n",
    "\n",
    "# Convert the lemmatized tokens back to text\n",
    "data['text'] = data['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))\n",
    "\n",
    "# Create a CountVectorizer to convert text into a matrix of token counts\n",
    "vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')\n",
    "X = vectorizer.fit_transform(data['text'])\n",
    "\n",
    "# Build LDA model\n",
    "lda_model = LatentDirichletAllocation(n_components=5, max_iter=20, random_state=42)\n",
    "lda_model.fit(X)\n",
    "\n",
    "# Print the top words for each topic\n",
    "feature_names = vectorizer.get_feature_names_out()\n",
    "print(\"Top words for each topic:\")\n",
    "for topic_idx, topic in enumerate(lda_model.components_):\n",
    "    print(f\"Topic {topic_idx + 1}:\")\n",
    "    print(\" \".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21f0b329",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
