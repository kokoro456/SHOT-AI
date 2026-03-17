package com.shot.app.viewmodel

import androidx.camera.view.PreviewView
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.shot.core.model.CourtDetectionResult
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject

/**
 * ViewModel that orchestrates the court detection pipeline:
 * Camera frame → ML detection → Homography → Projection → UI state
 */
@HiltViewModel
class CameraViewModel @Inject constructor(
    // TODO: Inject CourtKeypointDetector, HomographyCalculator, etc.
) : ViewModel() {

    private val _detectionResult = MutableStateFlow(CourtDetectionResult.EMPTY)
    val detectionResult: StateFlow<CourtDetectionResult> = _detectionResult.asStateFlow()

    private val _isDebugMode = MutableStateFlow(false)
    val isDebugMode: StateFlow<Boolean> = _isDebugMode.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    /**
     * Bind camera to the PreviewView and start frame analysis.
     * Called from the Compose UI when the PreviewView is created.
     */
    fun bindCamera(previewView: PreviewView) {
        // TODO: Implement CameraX binding with ImageAnalysis
        // 1. Get ProcessCameraProvider
        // 2. Create Preview use case → bind to previewView
        // 3. Create ImageAnalysis use case → STRATEGY_KEEP_ONLY_LATEST
        // 4. Set analyzer that calls processFrame()
        // 5. Bind to lifecycle
    }

    /**
     * Process a single camera frame through the detection pipeline.
     * Called from the ImageAnalysis analyzer on each frame.
     *
     * Pipeline:
     * 1. ML model inference → detected keypoints
     * 2. Filter by confidence threshold (>= 0.7)
     * 3. If >= 6 reliable keypoints → compute homography
     * 4. Validate homography (reprojection error < 5px)
     * 5. Temporal smoothing (EMA, alpha=0.7)
     * 6. Project all 16 keypoints → update UI state
     */
    private fun processFrame(/* imageProxy: ImageProxy */) {
        // TODO: Implement full pipeline
        // This will be connected in Step 10 (Pipeline Integration)
    }

    fun toggleDebugMode() {
        _isDebugMode.value = !_isDebugMode.value
    }

    fun toggleRecording() {
        _isRecording.value = !_isRecording.value
        // TODO: Start/stop CameraX VideoCapture
    }
}
