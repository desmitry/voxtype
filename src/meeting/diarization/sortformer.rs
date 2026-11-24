//! Sortformer diarization: NVIDIA's Nemotron-3 Diarization (Streaming
//! Sortformer v3, up to 8 speakers) through parakeet-rs.
//!
//! ECAPA clustering embeds a few seconds at a time and guesses who that was.
//! Sortformer instead scores every speaker for every 10 ms frame, so two
//! people talking over each other stay two people, and its speaker cache
//! keeps the same speaker IDs from one chunk to the next. Each transcript
//! segment goes to the speaker most active while it was said.
//!
//! Run behind [`super::hybrid::HybridDiarizer`], so the microphone is always
//! You and only the call (or a single mixed track) is diarized.

use super::{DiarizedSegment, Diarizer, SpeakerId};
use crate::meeting::data::AudioSource;
use crate::meeting::TranscriptSegment;
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
use parakeet_rs::ExecutionConfig;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Mean activity under which nobody counts as speaking in a segment. Such a
/// segment is left `Unknown`, and the hybrid wrapper gives it to whoever
/// spoke last.
const MIN_ACTIVITY: f32 = 0.2;

/// Streaming Sortformer over one track of a meeting.
pub struct SortformerDiarizer {
    model: Mutex<Sortformer>,
}

impl SortformerDiarizer {
    /// Where the model lives unless `[meeting.diarization] model_path` says otherwise.
    pub fn default_model_path() -> PathBuf {
        crate::config::Config::models_dir().join(crate::setup::model::SORTFORMER_MODEL_FILENAME)
    }

    /// Load the model at `path` into the shared ONNX Runtime environment.
    pub fn load(path: &Path) -> Result<Self, String> {
        let env = crate::ort_env::env().map_err(|e| e.to_string())?;
        let exec = ExecutionConfig::new().with_environment(env.clone());
        let model = Sortformer::with_config(path, Some(exec), DiarizationConfig::default())
            .map_err(|e| e.to_string())?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

/// The speaker most active over frames `[from, to)` of `preds`
/// (`frames x speakers`), or `None` when nobody reaches [`MIN_ACTIVITY`].
fn dominant_speaker(preds: &ndarray::Array2<f32>, from: usize, to: usize) -> Option<usize> {
    let to = to.min(preds.nrows());
    if from >= to {
        return None;
    }
    let window = preds.slice(ndarray::s![from..to, ..]);
    let n = (to - from) as f32;
    window
        .columns()
        .into_iter()
        .map(|c| c.sum() / n)
        .enumerate()
        .filter(|(_, mean)| *mean >= MIN_ACTIVITY)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(spk, _)| spk)
}

impl Diarizer for SortformerDiarizer {
    fn diarize(
        &self,
        samples: &[f32],
        _source: AudioSource,
        chunk_start_ms: u64,
        transcript_segments: &[TranscriptSegment],
    ) -> Vec<DiarizedSegment> {
        // Every chunk goes through the model, even one with no text, so the
        // speaker cache keeps following the call.
        let raw = match self.model.lock() {
            Ok(mut model) => model.diarize_chunk_raw(samples),
            Err(_) => {
                tracing::warn!("Sortformer state lock poisoned");
                return unknown(transcript_segments);
            }
        };
        let raw = match raw {
            Ok(raw) if raw.num_valid_frames > 0 => raw,
            Ok(_) => return unknown(transcript_segments),
            Err(e) => {
                tracing::warn!("Sortformer diarization failed: {}", e);
                return unknown(transcript_segments);
            }
        };

        let chunk_ms = samples.len() as f64 * 1000.0 / 16_000.0;
        let frame_ms = chunk_ms / raw.num_valid_frames as f64;
        let frame = |ms: u64| (ms.saturating_sub(chunk_start_ms) as f64 / frame_ms) as usize;

        transcript_segments
            .iter()
            .map(|seg| {
                let spk = dominant_speaker(
                    &raw.predictions,
                    frame(seg.start_ms),
                    frame(seg.end_ms).min(raw.num_valid_frames),
                );
                DiarizedSegment {
                    speaker: spk.map_or(SpeakerId::Unknown, |n| SpeakerId::Auto(n as u32)),
                    start_ms: seg.start_ms,
                    end_ms: seg.end_ms,
                    text: seg.text.clone(),
                    confidence: if spk.is_some() { 0.9 } else { 0.0 },
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "sortformer"
    }
}

fn unknown(segments: &[TranscriptSegment]) -> Vec<DiarizedSegment> {
    segments
        .iter()
        .map(|seg| DiarizedSegment {
            speaker: SpeakerId::Unknown,
            start_ms: seg.start_ms,
            end_ms: seg.end_ms,
            text: seg.text.clone(),
            confidence: 0.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn the_most_active_speaker_wins_even_while_another_talks_over_them() {
        // Speaker 1 talks throughout; speaker 0 interrupts for half of it.
        let preds = array![[0.9, 0.8], [0.9, 0.9], [0.1, 0.9], [0.0, 0.9]];
        assert_eq!(dominant_speaker(&preds, 0, 4), Some(1));
        assert_eq!(dominant_speaker(&preds, 0, 2), Some(0));
    }

    #[test]
    fn silence_and_empty_ranges_have_no_speaker() {
        let preds = array![[0.05, 0.1], [0.0, 0.1]];
        assert_eq!(dominant_speaker(&preds, 0, 2), None);
        assert_eq!(dominant_speaker(&preds, 2, 2), None);
        assert_eq!(dominant_speaker(&preds, 5, 9), None);
    }
}
