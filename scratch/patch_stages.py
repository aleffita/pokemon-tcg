import pickle
import numpy as np
import hashlib

paths = [
    'experiments/curriculum_v1/stage1/curriculum_v1_stage1.pkl',
    'experiments/curriculum_v1/stage2/curriculum_v1_stage2.pkl',
    'experiments/curriculum_v1/stage3/curriculum_v1_stage3.pkl'
]

for path in paths:
    print(f"Patching {path}...")
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
    print("Done.\n")
