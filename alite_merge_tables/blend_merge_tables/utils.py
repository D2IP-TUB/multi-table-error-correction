import re


def tokenize_text(text):
    """
    Original function provided by Mahdi Esmailoghli within the COCOA GitHub repository.
    """

    if text is None:
        return ''

    text = text.lower()

    stopwords = ['a', 'the', 'of', 'on', 'in', 'an', 'and', 'is', 'at', 'are', 'as', 'be', 'but', 'by', 'for', 'it', 'no', 'not',
                 'or', 'such', 'that', 'their', 'there', 'these', 'to', 'was', 'with', 'they', 'will', 'v', 've', 'd']

    cleaned = re.sub(r'[\W_]+', ' ', text.encode('ascii', 'ignore').decode('ascii'))
    feature_one = re.sub(r' +', ' ', cleaned).strip()

    for x in stopwords:
        feature_one = feature_one.replace(f' {x} ', ' ')
        if feature_one.startswith(f'{x} '):
            feature_one = feature_one[len(f'{x} '):]
        if feature_one.endswith(f' {x}'):
            feature_one = feature_one[:-len(f' {x}')]

    return feature_one

