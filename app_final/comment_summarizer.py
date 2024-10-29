import re
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

# Явне завантаження необхідних ресурсів NLTK
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab")


def clean_text_advanced(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z.,\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s,.]", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\S{40,}", "", text)
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    text = re.sub(url_pattern, "", text)
    text = re.sub(r"(\b\w+\b)(\s+\1)+", r"\1", text)
    text = re.sub(r"(\b.+?\b)(\s*\1)+", r"\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"(.)(.)\1\2{2,}", r"\1\2", text)
    text = re.sub(r"(ha|ah|mu|lol){2,}", "", text)
    text = re.sub(r"(\b\w+\b)(\s+\1){2,}", r"\1", text)
    text = re.sub(r"\b(\w+\b(?:\s+\w+\b){0,5})\s*(\1\s*)+", r"\1", text)
    text = re.sub(r"\b(\w+)\s+\1\s+\1(?:\s+\1)+", r"\1", text)
    text = re.sub(
        r"(\b(?:ha|ah|lol|mu|muah)+\b(?:\s*\b(?:ha|ah|lol|mu|muah)+\b){2,})",
        r"\1",
        text,
    )
    return text


class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def preprocess_text(self, text):
        sentences = sent_tokenize(text)
        word_tokens = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalnum() and w not in self.stop_words]
            word_tokens.extend(words)
        return sentences, word_tokens

    def get_sentence_scores(self, sentences, word_tokens):
        freq_dist = FreqDist(word_tokens)
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalnum()]
            score = sum(
                [freq_dist[word] for word in words if word not in self.stop_words]
            )
            sentence_scores[i] = score / len(words) if words else 0
        return sentence_scores

    def summarize(self, text, target_length=512):
        text = clean_text_advanced(text)
        sentences, word_tokens = self.preprocess_text(text)
        sentence_scores = self.get_sentence_scores(sentences, word_tokens)
        sorted_sentences = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )

        selected_sentences = []
        current_length = 0
        for sentence_idx, _ in sorted_sentences:
            sentence = sentences[sentence_idx]
            sentence_length = len(sentence)
            if current_length + sentence_length <= target_length:
                selected_sentences.append((sentence_idx, sentence))
                current_length += sentence_length
            else:
                break

        summary = " ".join([sentence for _, sentence in sorted(selected_sentences)])
        return summary


summarizer = TextSummarizer()


def main():
    summarizer = TextSummarizer()
    while True:
        comment = input("Введіть ваш коментар (або 'q' для виходу): ")
        if comment.lower() == "q":
            break
        summary = summarizer.summarize(comment)
        print("\nОброблений та сумаризований коментар:")
        print(summary)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
