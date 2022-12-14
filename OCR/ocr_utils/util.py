import pickle

def pickle_save(to_path, obj):
    with open(to_path, "wb") as f:
        pickle.dump(obj, f)
def pickle_load(from_path):
    with open(from_path, "rb") as f:
        data = pickle.load(f)
    return data