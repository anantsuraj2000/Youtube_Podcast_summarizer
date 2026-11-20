import json
import time
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from app.audio_extractor import extract_audio_from_youtube, get_video_duration, AudioExtractionError
from app.transcriber import transcribe_audio
from app.summarizer import extractive_summarize, chunk_transcript, SummarizationError
from app.chunking import chunk_transcript
from app.errors import SummarizationError
from app.summary_service import summarize


def run_pipeline_for_eval(url: str) -> dict:
    """Runs the pipeline on one video; returns the summary plus size/shape stats."""
    duration = get_video_duration(url)
    audio_path = extract_audio_from_youtube(url)
    transcript = transcribe_audio(audio_path)
    chunk_count = len(chunk_transcript(transcript))
    summary = extractive_summarize(transcript)
    return {
        "summary": summary,
        "duration_sec": duration,
        "transcript_chars": len(transcript),
        "chunk_count": chunk_count,
    }


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
            run = run_pipeline_for_eval(url)
            generated = run["summary"]
            elapsed = round(time.time() - start, 1)
        except Exception as e:  # known errors and unexpected ones (e.g. Groq/network): skip, keep going
            kind = "error" if isinstance(e, (AudioExtractionError, SummarizationError)) else "unexpected error"
            print(f"  Skipped due to {kind}: {e}")
            results.append({
                "video_url": url, "status": "failed", "error": str(e),
                "duration_sec": None, "transcript_chars": None, "chunk_count": None,
                "rouge1": None, "rouge2": None, "rougeL": None, "latency_sec": None,
            })
            continue

        rouge_scores = scorer.score(reference, generated)

        results.append({
            "video_url": url,
            "status": "success",
            "duration_sec": run["duration_sec"],
            "transcript_chars": run["transcript_chars"],
            "chunk_count": run["chunk_count"],
            "summary_words": len(generated.split()),
            "reference_words": len(reference.split()),
            "generated_summary": generated,
            "reference_summary": reference,
            "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
            "rouge2": round(rouge_scores["rouge2"].fmeasure, 4),
            "rougeL": round(rouge_scores["rougeL"].fmeasure, 4),
            "latency_sec": elapsed,
        })
        generated_summaries.append(generated)
        reference_summaries.append(reference)
        print(f"  ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f} | Time: {elapsed}s | "
              f"duration: {run['duration_sec']}s | transcript: {run['transcript_chars']} chars | "
              f"chunks: {run['chunk_count']}")

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
        print(f"Avg summary/reference words: "
              f"{success_df['summary_words'].mean():.0f} / {success_df['reference_words'].mean():.0f}")
        print(f"Avg chunks:    {success_df['chunk_count'].mean():.1f}")
        if len(success_df) >= 3:
            corr = success_df["chunk_count"].corr(success_df["rougeL"])
            print(f"Correlation chunk_count vs ROUGE-L: {corr:.2f}  "
                  "(negative would support 'longer videos score worse')")
    else:
        print("No videos were evaluated successfully — check errors in the CSV.")
    print(f"\nFull results saved to: {output_csv}")


if __name__ == "__main__":
    evaluate_test_set()