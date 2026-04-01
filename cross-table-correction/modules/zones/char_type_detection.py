import unicodedata


def get_all_char_types(strings):
    types = []
    for s in strings:
        for c in s:
            types.append(unicodedata.category(c))
    return types
