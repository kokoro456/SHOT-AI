package com.shot.app.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.findViewTreeLifecycleOwner
import com.shot.camera.CameraManager
import com.shot.court.CourtProjector
import com.shot.court.HomographyCalculator
import com.shot.court.TemporalSmoother
import com.shot.core.model.CourtDetectionResult
import com.shot.core.model.DetectionStatus
import com.shot.core.model.Keypoint
import com.shot.detection.CourtKeypointDetector
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import kotlin.math.sqrt

@HiltViewModel
class CameraViewModel @Inject constructor(
    application: Application
) : AndroidViewModel(application) {

    private val cameraManager = CameraManager(application)
    private var isCameraBound = false

    // ML pipeline components
    private val detector = CourtKeypointDetector(application)
    private val homographyCalculator = HomographyCalculator()
    private val projector = CourtProjector(homographyCalculator)
    private val smoother = TemporalSmoother()

    // Default keypoint positions (model outputs these for non-court scenes)
    // Captured from black/random input testing
    private val defaultPositions = floatArrayOf(
        0.29f, 0.53f,  // kp9
        0.52f, 0.53f,  // kp10
        0.66f, 0.55f,  // kp11
        0.29f, 0.71f,  // kp12
        0.36f, 0.70f,  // kp13
        0.56f, 0.70f,  // kp14
        0.77f, 0.71f,  // kp15
        0.73f, 0.61f   // kp16
    )

    private val _detectionResult = MutableStateFlow(CourtDetectionResult.EMPTY)
    val detectionResult: StateFlow<CourtDetectionResult> = _detectionResult.asStateFlow()

    private val _isDebugMode = MutableStateFlow(false)
    val isDebugMode: StateFlow<Boolean> = _isDebugMode.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    fun bindCamera(previewView: PreviewView) {
        if (isCameraBound) return
        val lifecycleOwner = previewView.findViewTreeLifecycleOwner() ?: return
        isCameraBound = true
        cameraManager.bind(
            previewView = previewView,
            lifecycleOwner = lifecycleOwner,
            onFrame = { imageProxy ->
                processFrame(imageProxy)
            }
        )
    }

    /**
     * Check if detected keypoints are too similar to the model's default output
     * (which appears for any non-court scene). If keypoints haven't moved from
     * defaults, we're not looking at a real court.
     *
     * Returns the mean distance from default positions (normalized 0-1 coords).
     */
    private fun distanceFromDefaults(keypoints: List<Keypoint>, imageWidth: Int, imageHeight: Int): Float {
        var totalDist = 0f
        for ((i, kp) in keypoints.withIndex()) {
            val normX = kp.x / imageWidth
            val normY = kp.y / imageHeight
            val defX = defaultPositions[i * 2]
            val defY = defaultPositions[i * 2 + 1]
            val dx = normX - defX
            val dy = normY - defY
            totalDist += sqrt(dx * dx + dy * dy)
        }
        return totalDist / keypoints.size
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val startTime = System.nanoTime()

            // Convert ImageProxy to Bitmap
            val bitmap = imageProxyToBitmap(imageProxy)
            val imageWidth = imageProxy.width
            val imageHeight = imageProxy.height

            // Step 1: Detect keypoints
            val keypoints = detector.detect(bitmap, imageWidth, imageHeight)
            bitmap.recycle()

            val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

            // Step 2: Check if keypoints differ from model's default output
            // Real court → keypoints move to actual court positions (distance > 0.03)
            // Non-court → keypoints stay at default positions (distance < 0.03)
            val defaultDist = distanceFromDefaults(keypoints, imageWidth, imageHeight)
            val isRealDetection = defaultDist > 0.03f

            Log.d("SHOT_DEBUG", "defaultDist=${"%.4f".format(defaultDist)} isReal=$isRealDetection")

            if (!isRealDetection) {
                _detectionResult.value = CourtDetectionResult(
                    detectedKeypoints = keypoints,
                    projectedKeypoints = emptyList(),
                    homographyMatrix = null,
                    reprojectionError = Float.MAX_VALUE,
                    inferenceTimeMs = inferenceTimeMs,
                    status = DetectionStatus.NOT_DETECTED
                )
                smoother.reset()
                return
            }

            // Step 3: Compute homography
            val rawH = homographyCalculator.computeHomography(keypoints)
            if (rawH == null) {
                _detectionResult.value = CourtDetectionResult(
                    detectedKeypoints = keypoints,
                    projectedKeypoints = emptyList(),
                    homographyMatrix = null,
                    reprojectionError = Float.MAX_VALUE,
                    inferenceTimeMs = inferenceTimeMs,
                    status = DetectionStatus.PARTIAL
                )
                return
            }

            // Step 4: Temporal smoothing
            val smoothedH = smoother.smooth(rawH, keypoints)

            // Step 5: Project all 16 keypoints
            val projectedKeypoints = projector.projectAllKeypoints(smoothedH)
            val reprojError = projector.computeReprojectionError(keypoints, smoothedH)

            Log.d("SHOT_DEBUG", "reproj=${"%.2f".format(reprojError)} projected=${projectedKeypoints.size}")

            // Step 6: Determine detection status
            val status = when {
                reprojError < 20f && projectedKeypoints.size >= 12 -> DetectionStatus.DETECTED
                reprojError < 50f && projectedKeypoints.size >= 8 -> DetectionStatus.PARTIAL
                else -> DetectionStatus.NOT_DETECTED
            }

            _detectionResult.value = CourtDetectionResult(
                detectedKeypoints = keypoints,
                projectedKeypoints = projectedKeypoints,
                homographyMatrix = smoothedH,
                reprojectionError = reprojError,
                inferenceTimeMs = inferenceTimeMs,
                status = status
            )
        } catch (e: Exception) {
            Log.e("SHOT_DEBUG", "processFrame error", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        // RGBA_8888 format (set in CameraManager)
        val buffer = imageProxy.planes[0].buffer
        val pixelStride = imageProxy.planes[0].pixelStride
        val rowStride = imageProxy.planes[0].rowStride
        val rowPadding = rowStride - pixelStride * imageProxy.width

        val bitmap = Bitmap.createBitmap(
            imageProxy.width + rowPadding / pixelStride,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        buffer.rewind()
        bitmap.copyPixelsFromBuffer(buffer)

        // Crop padding if needed
        return if (rowPadding > 0) {
            Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height)
        } else {
            bitmap
        }
    }

    fun toggleDebugMode() {
        _isDebugMode.value = !_isDebugMode.value
    }

    fun toggleRecording() {
        _isRecording.value = !_isRecording.value
    }

    override fun onCleared() {
        super.onCleared()
        cameraManager.shutdown()
        detector.close()
    }
}
