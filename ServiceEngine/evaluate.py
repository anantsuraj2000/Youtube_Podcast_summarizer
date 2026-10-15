import json
import time
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from app.audio_extractor import extract_audio_from_youtube, AudioExtractionError
from app.transcriber import transcribe_audio
from app.summarizer import extractive_summarize, SummarizationError


def run_pipeline_for_eval(url: str) -> str:
    """Runs the pipeline on one video and returns the generated summary."""
    wav_path = extract_audio_from_youtube(url)
    transcript = transcribe_audio(wav_path)
    summary = extractive_summarize(transcript)
    return summary


def evaluate_test_set(test_set_path: str = "test_set.json", output_csv: str = "evaluation_results.csv"):
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = []
    generated_summaries = []
    reference_summaries = []

    for i, item in enumerate(test_set):
        url = item["video_url"]
        reference = item["reference_summary"]
        print(f"\n[{i+1}/{len(test_set)}] Processing: {url}")

        start = time.time()
        try:
            generated = run_pipeline_for_eval(url)
            elapsed = round(time.time() - start, 1)
        except (AudioExtractionError, SummarizationError) as e:
            print(f"  Skipped due to error: {e}")
            results.append({
                "video_url": url, "status": "failed", "error": str(e),
                "rouge1": None, "rouge2": None, "rougeL": None, "latency_sec": None,
            })
            continue

        rouge_scores = scorer.score(reference, generated)

        results.append({
            "video_url": url,
            "status": "success",
            "generated_summary": generated,
            "reference_summary": reference,
            "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
            "rouge2": round(rouge_scores["rouge2"].fmeasure, 4),
            "rougeL": round(rouge_scores["rougeL"].fmeasure, 4),
            "latency_sec": elapsed,
        })
        generated_summaries.append(generated)
        reference_summaries.append(reference)
        print(f"  ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f} | Time: {elapsed}s")

    # BERTScore is computed in a batch across all successful summaries (more efficient)
    if generated_summaries:
        print("\nComputing BERTScore (this may take a minute on CPU)...")
        P, R, F1 = bert_score(generated_summaries, reference_summaries, lang="en", verbose=False)
        bert_f1_scores = F1.tolist()

        bert_idx = 0
        for r in results:
            if r["status"] == "success":
                r["bertscore_f1"] = round(bert_f1_scores[bert_idx], 4)
                bert_idx += 1

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    success_df = df[df["status"] == "success"]
    if len(success_df) > 0:
        print(f"Videos evaluated successfully: {len(success_df)}/{len(df)}")
        print(f"Avg ROUGE-1:   {success_df['rouge1'].mean():.4f}")
        print(f"Avg ROUGE-2:   {success_df['rouge2'].mean():.4f}")
        print(f"Avg ROUGE-L:   {success_df['rougeL'].mean():.4f}")
        print(f"Avg BERTScore: {success_df['bertscore_f1'].mean():.4f}")
        print(f"Avg latency:   {success_df['latency_sec'].mean():.1f}s")
    else:
        print("No videos were evaluated successfully — check errors in the CSV.")
    print(f"\nFull results saved to: {output_csv}")


if __name__ == "__main__":
    evaluate_test_set()