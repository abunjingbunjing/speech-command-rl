import numpy as np
import json
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import pickle

sys.path.append(str(Path(__file__).parent.parent))
from data_pipeline import COMMANDS, RESULTS_DIR, SEED

COMMAND_DESCRIPTIONS = {
    'yes':   'affirmative positive agreement confirmation approve correct',
    'no':    'negative denial refusal rejection disapprove incorrect',
    'up':    'upward direction increase raise higher ascend above top',
    'down':  'downward direction decrease lower reduce descend below bottom',
    'left':  'leftward direction turn rotate left side west port',
    'right': 'rightward direction turn rotate right side east starboard',
    'on':    'activate enable start power on switch begin run',
    'off':   'deactivate disable stop power off switch end halt',
    'stop':  'halt cease pause end terminate freeze interrupt command',
    'go':    'move proceed start continue forward action begin advance',
}

# Extended set — more varied phrases per class for better training
AUGMENTED_DESCRIPTIONS = {
    'yes':  ['affirmative positive agreement', 'confirm approve correct yes',
             'that is right agree', 'positive response confirmation'],
    'no':   ['negative denial refusal', 'disagree reject incorrect no',
             'that is wrong refuse', 'negative response rejection'],
    'up':   ['upward direction raise', 'increase higher ascend up',
             'go up move upward', 'above top elevation rise'],
    'down': ['downward direction lower', 'decrease reduce descend down',
             'go down move downward', 'below bottom drop fall'],
    'left': ['leftward direction turn left', 'rotate left side west',
             'move left go left', 'port side leftward turn'],
    'right':['rightward direction turn right', 'rotate right side east',
             'move right go right', 'starboard side rightward turn'],
    'on':   ['activate enable start', 'power on switch begin',
             'turn on run activate', 'enable start switch on'],
    'off':  ['deactivate disable stop', 'power off switch end',
             'turn off halt deactivate', 'disable stop switch off'],
    'stop': ['halt cease pause stop', 'end terminate freeze interrupt',
             'stop command halt now', 'cease action pause stop'],
    'go':   ['move proceed start go', 'continue forward advance begin',
             'go now move forward', 'start proceed action go'],
}


def build_nlp_dataset():
    """Build text + label pairs from augmented descriptions."""
    texts, labels = [], []
    for cmd, phrases in AUGMENTED_DESCRIPTIONS.items():
        for phrase in phrases:
            texts.append(phrase)
            labels.append(cmd)
    return texts, labels


def build_nlp_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ('clf',   LogisticRegression(max_iter=500, random_state=SEED,
                                     C=1.0, solver='lbfgs',
                                     multi_class='multinomial'))
    ])


if __name__ == "__main__":
    print("=== NLP Component: Command Label Classifier ===\n")

    # --- Build dataset ---
    texts, labels = build_nlp_dataset()
    print(f"Training samples: {len(texts)}")
    print(f"Classes         : {sorted(set(labels))}\n")

    # --- Train pipeline ---
    nlp_pipeline = build_nlp_pipeline()
    nlp_pipeline.fit(texts, labels)

    # --- Cross-validation (more reliable than a single train/test split) ---
    cv_scores = cross_val_score(
        build_nlp_pipeline(), texts, labels, cv=4, scoring='f1_macro'
    )
    print(f"Cross-val macro F1: {cv_scores.mean():.3f} "
          f"(+/- {cv_scores.std():.3f})")

    # --- Evaluate on training set (sanity check) ---
    train_preds = nlp_pipeline.predict(texts)
    print("\n=== Classification Report (train) ===")
    print(classification_report(labels, train_preds, target_names=sorted(set(labels))))

    # --- Test with unseen phrases ---
    test_phrases = [
        ('please confirm yes agree',  'yes'),
        ('turn off disable power',    'off'),
        ('move forward go now',       'go'),
        ('halt stop cease action',    'stop'),
        ('raise increase go higher',  'up'),
    ]
    print("=== Unseen phrase predictions ===")
    all_correct = 0
    for phrase, expected in test_phrases:
        pred  = nlp_pipeline.predict([phrase])[0]
        proba = nlp_pipeline.predict_proba([phrase]).max()
        match = "✓" if pred == expected else "✗"
        print(f"  {match} '{phrase}'")
        print(f"    Expected: {expected} | Predicted: {pred} | Confidence: {proba:.3f}")
        all_correct += int(pred == expected)
    print(f"\nUnseen accuracy: {all_correct}/{len(test_phrases)}")

    # --- Save trained model for reuse ---
    model_save_path = RESULTS_DIR / "nlp_pipeline.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(nlp_pipeline, f)
    print(f"\nNLP model saved to {model_save_path}")

    # --- Save NLP metrics as JSON ---
    nlp_results = {
        "cv_macro_f1_mean" : float(cv_scores.mean()),
        "cv_macro_f1_std"  : float(cv_scores.std()),
        "train_samples"    : len(texts),
        "n_classes"        : len(set(labels)),
        "unseen_accuracy"  : f"{all_correct}/{len(test_phrases)}",
    }
    with open(RESULTS_DIR / "nlp_results.json", "w") as f:
        json.dump(nlp_results, f, indent=2)

    print(f"NLP results saved to {RESULTS_DIR}/nlp_results.json")
    print("\nnlp.py complete.")
