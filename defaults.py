

def default_preprocessor(data):
    return data

def default_preprocessor_inverse(data):
    return data

def default_postprocessor(adv_data, base_data):
    return adv_data

def default_target_picker(surrogate_model, surrogate_test_dataset):
    return dict()