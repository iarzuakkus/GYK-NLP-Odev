import json
from predict import predict_summary
import evaluate
import os

def load_test_data(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data

def run_rouge_evaluation(test_data, save_path=None):
    references = []
    predictions = []
    results = []

    print("\nComparative Results:\n" + "="*60)

    for idx, item in enumerate(test_data):
        article = item["article"]
        reference = item["summary"]
        prediction = predict_summary(article)

        references.append(reference)
        predictions.append(prediction)

        results.append({
            "index": idx,
            "article": article,
            "reference_summary": reference,
            "predicted_summary": prediction
        })

        print(f"\n[{idx+1}]")
        print(f"Reference : {reference}")
        print(f"Predicted : {prediction}")

    # ROUGE hesapla
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    print("\n\n ROUGE Scores:\n" + "-"*20)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "comparisons": results,
                "rouge_scores": scores
            }, f, indent=2)
        print(f"\n Results saved to: {save_path}")

    return scores

if __name__ == "__main__":
    test_file = "data/cnn_dailymail_test.json"
    save_file = "outputs/test_results_m2.json"

    test_samples = load_test_data(test_file, limit=10)
    print(f"\nLoaded {len(test_samples)} test examples.")
    run_rouge_evaluation(test_samples, save_path=save_file)
