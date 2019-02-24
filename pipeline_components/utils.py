def update_dictionary(dictionary, update):
    new_keys = set(update.keys()).difference(set(dictionary.keys()))
    if len(new_keys) > 0:
        raise Exception('Trying to update dictionary with non-existing keys: {}'.format(new_keys))

    return dict(list(dictionary.items()) + list(update.items()))
