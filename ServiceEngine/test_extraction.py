from app.audio_extractor import extract_audio_from_youtube
from app.transcriber import transcribe_audio
from app.summarizer import extractive_summarize
from app.refiner import refine_summary


#url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
url = "https://www.youtube.com/watch?v=FRI8QjViK7s"
wav_path = extract_audio_from_youtube(url)
print(f"Audio saved to: {wav_path}")

transcript = transcribe_audio(wav_path)
print(f"\nTranscript length: {len(transcript)} characters")

summary = extractive_summarize(transcript)
refined = refine_summary(summary)
# print("\n--- SUMMARY ---\n")
# print(summary)
print("\n--- SUMMARY ---\n", refined["summary"])
print("\n--- ENTITIES ---\n", refined["entities"])