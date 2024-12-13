from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import nltk
import syllapy
from nltk.corpus import stopwords, cmudict
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from concurrent.futures import ThreadPoolExecutor
import re
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')
personal_pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'us', 'him', 'her', 'them', 'my', 'your', 'his', 'hers', 'our', 'their', 'mine', 'yours', 'ours', 'theirs']

class LoadExportData:
    def __init__(self, file):
        self.data = pd.read_excel(file)
        self.final_data = pd.DataFrame()

    def export(self, df, name):
        self.final_data = df
        df.to_csv(name)

class ExtractData(LoadExportData):
    def __init__(self, file):
        self.data = LoadExportData(file).data
        
    def get_article_data(self, url):
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            title = soup.find('h1', class_='entry-title')  
            text = soup.find('div', class_='td-pb-span8 td-main-content')
            
            if title and text:
                title = title.text.strip()
                text = text.text.strip()
                filename = title.replace(" ", "_").replace("/", "_").replace(":", "") + '.txt'
                os.makedirs('article_data', exist_ok=True)
                file_path = f'article_data/{filename}'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {title}\n\n")
                    f.write(text)
                    print(f"Saved article '{title}' to {file_path}")
                
                return file_path
            return None
        except Exception as e:
            print(f"Failed to retrieve article from {url}. Error: {e}")
            return None

    def save_articles(self):
        with ThreadPoolExecutor() as executor:
            file_paths = list(executor.map(self.get_article_data, self.data['URL']))
        
        self.data['title_file'] = file_paths
        export_obj = LoadExportData('Input.xlsx')
        export_obj.export(self.data, 'articles_with_titles.csv')


class TextAnalysis:
    def __init__(self, df):
        self.df = df
        self.stop_words = set(stopwords.words('english'))
        self.pron_dict = cmudict.dict()
        
    def read_text_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def find_words(self, text, word_list):
        words_in_text = text.split()
        return [word for word in words_in_text if word.lower() in word_list]

    def remove_stopwords(self, text):
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalpha() and word not in self.stop_words]
        return filtered_words

    def count_complex_words_from_list(self, filtered_words):
        complex_words = [word for word in filtered_words if syllapy.count(word) >= 3]
        return complex_words

    def syllable_count(self, word):
        word = word.lower()
        if word in self.pron_dict:
            return max([len(list(y for y in x if y[-1].isdigit())) for x in self.pron_dict[word]])
        else:
            return len([char for char in word if char in "aeiou"])

    def calculate_word_count(self, text):
        return len(text.split())

    def calculate_fog_index(self, row):
        sentences = re.split(r'[.!?]', row['text'])
        sentence_count = len([s for s in sentences if s.strip() != ''])
        word_count = row['word_count']
        complex_word_count = row['complex_word_count']

        if sentence_count == 0 or word_count == 0:
            return 0
        return 0.4 * ((word_count / sentence_count) + 100 * (complex_word_count / word_count))

    def calculate_avg_words_per_sentence(self, row):
        sentences = row['text']
        sentence_list = nltk.sent_tokenize(sentences)
        word_count = sum(len(nltk.word_tokenize(sentence)) for sentence in sentence_list)
        sentence_count = len(sentence_list)
        
        if sentence_count == 0:
            return 0
        return int(word_count / sentence_count)

    def calculate_complex_percentage(self, row):
        complex_word_count = 0
        words = row['filtered_words']
        
        for word in words:
            if self.syllable_count(word) > 2:
                complex_word_count += 1
        
        word_count = row['word_count']
        
        if word_count == 0:
            return 0
        return (complex_word_count / word_count) * 100
    def calculate_subjectivity(self, text):
        blob = TextBlob(text)
        return blob.sentiment.subjectivity  # Subjectivity score is between 0 and 1

    # Function to count personal pronouns
    def count_personal_pronouns(self, text):
        words = word_tokenize(text.lower())  # Tokenize text and convert to lowercase
        pronouns_in_text = [word for word in words if word in personal_pronouns]
        return len(pronouns_in_text)
    def syllables_per_word(self, text):
        words = word_tokenize(text)
        syllable_count = sum([syllapy.count(word) for word in words])
        return syllable_count / len(words) if words else 0  # Avoid division by zero
    def calculate_polarity(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    # Function to calculate average sentence length
    def average_sentence_length(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        return len(words) / len(sentences) if sentences else 0  # Avoid division by zero

    def analyze_text(self):
        self.df['text'] = self.df['title_file'].apply(self.read_text_file)
        self.df['positive_words'] = self.df['text'].apply(lambda x: self.find_words(x, positive_words))
        self.df['negative_words'] = self.df['text'].apply(lambda x: self.find_words(x, negative_words))
        self.df['positive_word_count'] = self.df['positive_words'].apply(len)
        self.df['negative_word_count'] = self.df['negative_words'].apply(len)
        self.df['syllables_per_word'] = self.df['text'].apply(self.syllables_per_word)
        self.df['polarity_score'] = self.df['text'].apply(self.calculate_polarity)
        self.df['average_sentence_length'] = self.df['text'].apply(self.average_sentence_length)

        self.df['filtered_words'] = self.df['text'].apply(self.remove_stopwords)
        self.df['complex_words'] = self.df['filtered_words'].apply(self.count_complex_words_from_list)
        self.df['complex_word_count'] = self.df['complex_words'].apply(len)
        self.df['subjectivity_score'] = self.df['text'].apply(self.calculate_subjectivity)

        # Add personal pronoun count to the DataFrame
        self.df['personal_pronoun_count'] = self.df['text'].apply(self.count_personal_pronouns)
        self.df['word_count'] = self.df['text'].apply(self.calculate_word_count)
        self.df['avg_words_per_sentence'] = self.df.apply(self.calculate_avg_words_per_sentence, axis=1)
        self.df['fog_index'] = self.df.apply(self.calculate_fog_index, axis=1)
        self.df['percentage_complex_words'] = self.df.apply(self.calculate_complex_percentage, axis=1)


# Example usage
file = 'Input.xlsx'  # Path to your input file
extractor = ExtractData(file)
extractor.save_articles()

folder_path = 'MasterDictionary/'  # Path to your dictionary folder

with open(f'{folder_path}/negative-words.txt', 'r') as file:
    negative_words = file.read().splitlines()

with open(f'{folder_path}/positive-words.txt', 'r') as file:
    positive_words = file.read().splitlines()

# Perform text analysis
text_analysis = TextAnalysis(extractor.data)
text_analysis.analyze_text()

# Export the results
export_obj = LoadExportData('Input.xlsx')
export_obj.export(text_analysis.df, 'articles_with_analysis.csv')
