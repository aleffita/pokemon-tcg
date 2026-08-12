import pickle
import numpy as np
import hashlib

path = 'experiments/curriculum_v1/stage4/curriculum_v1_stage4.pkl'

with open(path, 'rb') as f:
    state = pickle.load(f)

if 'static_card_features' in state and state['static_card_features'] is not None:
    static_array = np.asarray(state['static_card_features'], dtype=np.float32)
    digest = hashlib.sha256(static_array.tobytes(order="C")).hexdigest()
    state['static_feature_contract']['sha256'] = digest
    state['static_feature_contract']['dtype'] = str(static_array.dtype)
    print(f"Patched static_feature_contract to float32. New SHA256: {digest}")

with open(path, 'wb') as f:
    pickle.dump(state, f)
