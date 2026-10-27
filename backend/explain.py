"""Explain a TF-IDF + linear model prediction: which words pushed the ticket towards the chosen class?

Used by the notebook (error analysis) AND the Flask API (the "keywords" in every response).

For a linear model the score of a class is simply
    score(class) = sum over words of  tfidf_value(word) * coefficient(class, word)  + intercept
so each word's CONTRIBUTION is tfidf_value * coefficient. Words with the biggest positive contribution are the ones
that "voted" for the class. This is exact for linear models, no approximation needed.
"""
import numpy as np


def top_keywords(pipeline, clean_text, class_name, k=6):
    """Return up to k (word, contribution) pairs that pushed `clean_text` towards `class_name`.

    pipeline   : sklearn Pipeline with steps "tfidf" (TfidfVectorizer) and "clf" (a linear classifier with coef_)
    clean_text : text already cleaned with text_utils.ticket_to_text
    """
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    row = vectorizer.transform([clean_text])            # 1 x vocabulary (sparse)
    class_index = list(clf.classes_).index(class_name)
    coefs = clf.coef_[class_index] if clf.coef_.shape[0] > 1 else clf.coef_[0]
    words = vectorizer.get_feature_names_out()

    cols = row.indices                                   # only the words that appear in this ticket
    contributions = row.data * coefs[cols]
    order = np.argsort(-contributions)
    result = []
    for i in order:
        if contributions[i] <= 0 or len(result) == k:
            break
        result.append((str(words[cols[i]]), round(float(contributions[i]), 3)))
    return result
