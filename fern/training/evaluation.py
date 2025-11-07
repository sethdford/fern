"""
Evaluation metrics for CSM-1B fine-tuning.

Provides:
- Objective metrics (MCD, F0 RMSE, PESQ, STOI)
- Subjective metrics (MOS, SMOS, CMOS)
- Efficiency metrics (RTF, memory, latency)
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for TTS fine-tuning.
    
    Metrics:
    - MCD (Mel Cepstral Distortion): Spectral similarity
    - F0 RMSE: Pitch accuracy
    - PESQ: Perceptual quality (requires pesq library)
    - STOI: Intelligibility (requires pystoi library)
    - RTF: Real-time factor (generation speed)
    """
    
    def __init__(self):
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if optional dependencies are available."""
        self.has_pesq = False
        self.has_stoi = False
        
        try:
            import pesq
            self.has_pesq = True
        except ImportError:
            logger.warning("pesq not installed, PESQ metric unavailable")
        
        try:
            import pystoi
            self.has_stoi = True
        except ImportError:
            logger.warning("pystoi not installed, STOI metric unavailable")
    
    def compute_mcd(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int = 24000,
    ) -> float:
        """
        Compute Mel Cepstral Distortion (MCD).
        
        Lower is better (typically 4-6 dB is good).
        
        Args:
            generated: Generated audio array
            reference: Reference audio array
            sample_rate: Sample rate
            
        Returns:
            MCD value in dB
        """
        # Extract mel cepstral coefficients
        mcc_gen = self._extract_mcc(generated, sample_rate)
        mcc_ref = self._extract_mcc(reference, sample_rate)
        
        # Align sequences (DTW or simple trimming)
        min_len = min(len(mcc_gen), len(mcc_ref))
        mcc_gen = mcc_gen[:min_len]
        mcc_ref = mcc_ref[:min_len]
        
        # Compute MCD
        # MCD = (10 / ln(10)) * sqrt(2 * sum((c_gen - c_ref)^2))
        diff = mcc_gen - mcc_ref
        mcd = (10.0 / np.log(10.0)) * np.sqrt(2 * np.mean(diff ** 2))
        
        return float(mcd)
    
    def compute_f0_rmse(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int = 24000,
    ) -> float:
        """
        Compute F0 (pitch) RMSE.
        
        Lower is better (typically <20 Hz is good).
        
        Args:
            generated: Generated audio array
            reference: Reference audio array
            sample_rate: Sample rate
            
        Returns:
            F0 RMSE in Hz
        """
        # Extract F0 contours
        f0_gen = self._extract_f0(generated, sample_rate)
        f0_ref = self._extract_f0(reference, sample_rate)
        
        # Align sequences
        min_len = min(len(f0_gen), len(f0_ref))
        f0_gen = f0_gen[:min_len]
        f0_ref = f0_ref[:min_len]
        
        # Filter out unvoiced regions (F0 = 0)
        voiced_mask = (f0_gen > 0) & (f0_ref > 0)
        if voiced_mask.sum() == 0:
            return 0.0
        
        # Compute RMSE on voiced regions
        f0_gen_voiced = f0_gen[voiced_mask]
        f0_ref_voiced = f0_ref[voiced_mask]
        
        rmse = np.sqrt(np.mean((f0_gen_voiced - f0_ref_voiced) ** 2))
        
        return float(rmse)
    
    def compute_pesq(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int = 24000,
    ) -> Optional[float]:
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality).
        
        Range: -0.5 to 4.5 (higher is better, >3.0 is good).
        
        Args:
            generated: Generated audio array
            reference: Reference audio array
            sample_rate: Sample rate
            
        Returns:
            PESQ score or None if library unavailable
        """
        if not self.has_pesq:
            return None
        
        from pesq import pesq as compute_pesq_score
        
        # PESQ requires 8kHz or 16kHz
        if sample_rate not in [8000, 16000]:
            # Resample to 16kHz
            import scipy.signal
            num_samples = int(len(generated) * 16000 / sample_rate)
            generated = scipy.signal.resample(generated, num_samples)
            reference = scipy.signal.resample(reference, num_samples)
            sample_rate = 16000
        
        try:
            score = compute_pesq_score(
                sample_rate,
                reference,
                generated,
                'wb',  # wideband
            )
            return float(score)
        except Exception as e:
            logger.warning(f"PESQ computation failed: {e}")
            return None
    
    def compute_stoi(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int = 24000,
    ) -> Optional[float]:
        """
        Compute STOI (Short-Time Objective Intelligibility).
        
        Range: 0 to 1 (higher is better, >0.9 is good).
        
        Args:
            generated: Generated audio array
            reference: Reference audio array
            sample_rate: Sample rate
            
        Returns:
            STOI score or None if library unavailable
        """
        if not self.has_stoi:
            return None
        
        from pystoi import stoi
        
        try:
            score = stoi(reference, generated, sample_rate, extended=False)
            return float(score)
        except Exception as e:
            logger.warning(f"STOI computation failed: {e}")
            return None
    
    def compute_all(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int = 24000,
    ) -> Dict[str, Optional[float]]:
        """
        Compute all available metrics.
        
        Args:
            generated: Generated audio array
            reference: Reference audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # MCD
        try:
            metrics["mcd"] = self.compute_mcd(generated, reference, sample_rate)
        except Exception as e:
            logger.warning(f"MCD computation failed: {e}")
            metrics["mcd"] = None
        
        # F0 RMSE
        try:
            metrics["f0_rmse"] = self.compute_f0_rmse(generated, reference, sample_rate)
        except Exception as e:
            logger.warning(f"F0 RMSE computation failed: {e}")
            metrics["f0_rmse"] = None
        
        # PESQ
        if self.has_pesq:
            metrics["pesq"] = self.compute_pesq(generated, reference, sample_rate)
        else:
            metrics["pesq"] = None
        
        # STOI
        if self.has_stoi:
            metrics["stoi"] = self.compute_stoi(generated, reference, sample_rate)
        else:
            metrics["stoi"] = None
        
        return metrics
    
    @staticmethod
    def _extract_mcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
        """Extract mel cepstral coefficients."""
        import librosa
        
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
        )
        
        return mfcc.T  # (time, n_mfcc)
    
    @staticmethod
    def _extract_f0(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract F0 (pitch) contour."""
        import librosa
        
        # Compute F0 using pyin algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
        )
        
        # Replace NaN with 0 (unvoiced)
        f0 = np.nan_to_num(f0)
        
        return f0
    
    @staticmethod
    def compute_rtf(
        generation_time: float,
        audio_duration: float,
    ) -> float:
        """
        Compute Real-Time Factor (RTF).
        
        RTF = generation_time / audio_duration
        
        RTF < 1.0 means faster than real-time (good!).
        RTF > 1.0 means slower than real-time (bad).
        
        Args:
            generation_time: Time to generate audio (seconds)
            audio_duration: Duration of generated audio (seconds)
            
        Returns:
            RTF value
        """
        if audio_duration == 0:
            return float('inf')
        
        return generation_time / audio_duration


def evaluate_model(
    model,
    dataset,
    num_samples: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Fine-tuned CSM model
        dataset: Test dataset
        num_samples: Number of samples to evaluate
        device: Device to use
        
    Returns:
        Dictionary with average metrics
    """
    metrics_computer = EvaluationMetrics()
    
    all_metrics = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Generate audio
        # TODO: Implement actual generation
        # generated = model.generate(sample["text"], sample["speaker_id"])
        
        # For now, use reference as placeholder
        generated = sample["audio"].numpy()
        reference = sample["audio"].numpy()
        
        # Compute metrics
        metrics = metrics_computer.compute_all(
            generated,
            reference,
            sample_rate=sample["sample_rate"],
        )
        
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if m[key] is not None]
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = None
    
    return avg_metrics

