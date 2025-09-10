from faster_whisper import WhisperModel, BatchedInferencePipeline
from pydub import AudioSegment
import os
import time
import tempfile
import concurrent.futures
import shutil

# --- Configuration ---
SOURCE_STEREO_FILE_PATH = 'files/10_min_clear.wav'

NUM_CONCURRENT_FILES = 10
BATCH_SIZE = 16
PRINT_TRANSCRIPT_SAMPLE = True

BEAM_SIZE = 5 # 5 is good for quality. Reducing will make it faster, but will degrade quality
MODEL_SIZE = "large-v2"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

CUSTOMER_CHANNEL_INDEX = 0
AGENT_CHANNEL_INDEX = 1



# --- Helper Functions ---

def format_conversation_faster_whisper(batch_results):
    """
    Merges word timestamps from faster-whisper's output into a single,
    chronologically sorted conversational transcript.
    (This function remains the same and works for both methods)
    """
    all_words = []
    if not batch_results or len(batch_results) < 2:
        return "[Error: Not enough channel data to format conversation]"

    # Unpack results: results are tuples of (segments_generator, info)
    processed_results = [list(res[0]) for res in batch_results]

    for i, segments in enumerate(processed_results):
        speaker = "Customer" if i == CUSTOMER_CHANNEL_INDEX else "Agent"
        if not segments: continue
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    all_words.append({
                        'start': word.start, 'end': word.end,
                        'word': word.word.strip(), 'speaker': speaker
                    })

    if not all_words: return "[Transcription resulted in no words]"
    all_words.sort(key=lambda x: x['start'])

    full_transcript = []
    current_speaker = all_words[0]['speaker']
    current_utterance = [all_words[0]['word']]

    for word_info in all_words[1:]:
        if word_info['speaker'] != current_speaker:
            full_transcript.append(f"{current_speaker}: {' '.join(current_utterance)}")
            current_speaker = word_info['speaker']
            current_utterance = [word_info['word']]
        else:
            current_utterance.append(word_info['word'])

    full_transcript.append(f"{current_speaker}: {' '.join(current_utterance)}")
    return "\n".join(full_transcript)


def prepare_benchmark_files(source_path, num_files, temp_dir):
    """Creates copies of the source file for the benchmark."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    filenames = []
    for i in range(num_files):
        new_path = os.path.join(temp_dir, f"source_file_{i}.wav")
        shutil.copy(source_path, new_path)
        filenames.append(new_path)

    audio = AudioSegment.from_wav(filenames[0])
    duration_seconds = len(audio) / 1000.0

    print(f"\nPrepared {num_files} copies of '{os.path.basename(source_path)}'.")
    print(f"Each file is {duration_seconds:.2f} seconds long.")
    print(f"Total audio to process: {num_files * duration_seconds / 60:.2f} minutes.")

    return filenames, duration_seconds


# --- Core Task Logic ---

def process_single_stereo_file(model_or_pipeline, stereo_filepath, temp_dir_for_file, use_batching):
    """
    Transcribes a single stereo file. Can use either the standard model
    or the batched pipeline, based on the `use_batching` flag.
    """
    stereo_audio = AudioSegment.from_wav(stereo_filepath)
    mono_channels = stereo_audio.split_to_mono()

    file_basename = os.path.splitext(os.path.basename(stereo_filepath))[0]
    mono_filepaths = [
        os.path.join(temp_dir_for_file, f'{file_basename}_left.wav'),
        os.path.join(temp_dir_for_file, f'{file_basename}_right.wav')
    ]
    mono_channels[CUSTOMER_CHANNEL_INDEX].export(mono_filepaths[CUSTOMER_CHANNEL_INDEX], format="wav")
    mono_channels[AGENT_CHANNEL_INDEX].export(mono_filepaths[AGENT_CHANNEL_INDEX], format="wav")

    # Common transcription options
    transcribe_options = {
        "word_timestamps": True,
        "language": "en",
        "beam_size": BEAM_SIZE,
        "vad_filter": True,
        "vad_parameters": dict(min_silence_duration_ms=500)
    }

    # Add the batch_size argument only when using the batched pipeline
    if use_batching:
        transcribe_options["batch_size"] = BATCH_SIZE

    ordered_results = [None] * 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_index = {
            executor.submit(
                model_or_pipeline.transcribe, path, **transcribe_options
            ): i for i, path in enumerate(mono_filepaths)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            ordered_results[index] = future.result()

    return format_conversation_faster_whisper(ordered_results)


# --- Benchmark Functions ---

def run_benchmark(name, model_or_pipeline, use_batching, stereo_filepaths, total_audio_duration_seconds):
    """A generic function to run a benchmark."""
    print("\n" + "=" * 80)
    print(f"--- BENCHMARK: {name} ---")
    print(f"Processing {len(stereo_filepaths)} files concurrently.")
    print("=" * 80)

    start_time = time.perf_counter()
    transcripts = []

    with tempfile.TemporaryDirectory() as temp_dir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(stereo_filepaths)) as executor:
            future_to_path = {
                executor.submit(process_single_stereo_file, model_or_pipeline, path, temp_dir, use_batching): path
                for path in stereo_filepaths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    transcript = future.result()
                    transcripts.append(transcript)
                    print(f"Finished processing: {os.path.basename(path)}")
                except Exception as exc:
                    print(f"{os.path.basename(path)} generated an exception: {exc}")
                    transcripts.append(f"[ERROR: {exc}]")

    end_time = time.perf_counter()
    total_time_seconds = end_time - start_time
    audio_mins_per_proc_min = total_audio_duration_seconds / total_time_seconds

    print("-" * 80)
    print(f"Benchmark '{name}' Finished")
    print(f"Total time taken: {total_time_seconds:.2f} seconds")
    print(f"Minutes of audio processed per minute: {audio_mins_per_proc_min:.2f}x")
    print("-" * 80)

    if PRINT_TRANSCRIPT_SAMPLE and transcripts:
        print("\n--- Transcript Sample (First File) ---\n")
        print(transcripts[0])
        print("\n--------------------------------------\n")


def main():
    """
    Main function to load models and run the two parallel benchmarks.
    """
    print(f"Loading faster-whisper model '{MODEL_SIZE}'... (This happens only once)")
    # This is the standard model
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    # This is the pipeline for internal batching (old API)
    # The constructor does NOT take batch_size
    batched_model = BatchedInferencePipeline(model=model)
    print("Model and pipeline loaded and ready.")

    with tempfile.TemporaryDirectory() as temp_dir:
        stereo_filepaths, single_duration_sec = prepare_benchmark_files(
            SOURCE_STEREO_FILE_PATH, NUM_CONCURRENT_FILES, temp_dir
        )
        total_audio_duration_sec = len(stereo_filepaths) * single_duration_sec

        # --- Run the benchmarks ---

        # Benchmark 1: Concurrent processing, no internal batching
        #run_benchmark(
        #    "Concurrent (No Batching)", model, False,
        #    stereo_filepaths, total_audio_duration_sec
        #)

        # Benchmark 2: Concurrent processing, WITH internal batching per file
        run_benchmark(
            f"Concurrent With Internal Batching. Batch Size: {BATCH_SIZE}", batched_model, True,
            stereo_filepaths, total_audio_duration_sec
        )


if __name__ == "__main__":
    if not os.path.exists(SOURCE_STEREO_FILE_PATH):
        print(f"ERROR: Source audio file not found at '{SOURCE_STEREO_FILE_PATH}'")
        print("Please update the SOURCE_STEREO_FILE_PATH variable.")
    else:
        main()