package com.shot.app.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.findViewTreeLifecycleOwner
import com.shot.camera.CameraManager
import com.shot.court.CourtProjector
import com.shot.court.HomographyCalculator
import com.shot.court.HomographyValidator
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
import kotlin.math.abs
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
    private val validator = HomographyValidator()
    private val smoother = TemporalSmoother()

    // Default keypoint positions (model outputs these for non-court scenes)
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
     * Fast pre-check: is the image usable for court detection?
     * Rejects covered cameras, pitch-black, or uniform scenes.
     * Samples a 32x32 grid and checks grayscale variance.
     */
    private fun isImageUsable(bitmap: Bitmap): Boolean {
        val small = Bitmap.createScaledBitmap(bitmap, 32, 32, false)
        var sum = 0.0
        var sumSq = 0.0
        val total = 32 * 32
        for (y in 0 until 32) {
            for (x in 0 until 32) {
                val pixel = small.getPixel(x, y)
                val gray = (Color.red(pixel) * 0.299 + Color.green(pixel) * 0.587 + Color.blue(pixel) * 0.114)
                sum += gray
                sumSq += gray * gray
            }
        }
        small.recycle()
        val mean = sum / total
        val variance = (sumSq / total) - (mean * mean)
        return variance > 100.0 // uniform/dark images have very low variance
    }

    /**
     * Multi-criteria check: are these keypoints from a real court?
     */
    private fun isRealDetection(keypoints: List<Keypoint>, imageWidth: Int, imageHeight: Int): Boolean {
        if (keypoints.size < 6) return false

        // Criterion 1: Mean confidence must be reasonable
        val meanConf = keypoints.map { it.confidence }.average().toFloat()
        if (meanConf < 0.5f) return false

        // Criterion 2: Confidence shouldn't be too scattered (real courts are more uniform)
        val confValues = keypoints.map { it.confidence.toDouble() }
        val confMean = confValues.average()
        val confStd = sqrt(confValues.map { (it - confMean) * (it - confMean) }.average()).toFloat()
        if (confStd > 0.25f) return false

        // Criterion 3: Baseline should span reasonable width
        val kp12 = keypoints.find { it.id == 12 }
        val kp16 = keypoints.find { it.id == 16 }
        if (kp12 != null && kp16 != null) {
            val baselineWidth = abs(kp16.x - kp12.x) / imageWidth
            if (baselineWidth < 0.15f) return false
        }

        // Criterion 4: Distance from model's default output
        val defaultDist = distanceFromDefaults(keypoints, imageWidth, imageHeight)
        if (defaultDist < 0.02f) return false

        // Criterion 5: At least 6 keypoints with high confidence
        val reliableCount = keypoints.count { it.confidence > 0.7f }
        if (reliableCount < 6) return false

        return true
    }

    private fun distanceFromDefaults(keypoints: List<Keypoint>, imageWidth: Int, imageHeight: Int): Float {
        var totalDist = 0f
        var count = 0
        for (kp in keypoints) {
            val idx = kp.id - 9
            if (idx < 0 || idx >= 8) continue
            val normX = kp.x / imageWidth
            val normY = kp.y / imageHeight
            val defX = defaultPositions[idx * 2]
            val defY = defaultPositions[idx * 2 + 1]
            val dx = normX - defX
            val dy = normY - defY
            totalDist += sqrt(dx * dx + dy * dy)
            count++
        }
        return if (count > 0) totalDist / count else 0f
    }

    private fun emitNotDetected(keypoints: List<Keypoint>, inferenceTimeMs: Long) {
        _detectionResult.value = CourtDetectionResult(
            detectedKeypoints = keypoints,
            projectedKeypoints = emptyList(),
            homographyMatrix = null,
            reprojectionError = Float.MAX_VALUE,
            inferenceTimeMs = inferenceTimeMs,
            status = DetectionStatus.NOT_DETECTED
        )
        smoother.reset()
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val startTime = System.nanoTime()

            val bitmap = imageProxyToBitmap(imageProxy)
            val imageWidth = imageProxy.width
            val imageHeight = imageProxy.height

            // Pre-check: reject uniform/dark images before inference
            if (!isImageUsable(bitmap)) {
                bitmap.recycle()
                emitNotDetected(emptyList(), (System.nanoTime() - startTime) / 1_000_000)
                return
            }

            // Step 1: Detect keypoints
            val keypoints = detector.detect(bitmap, imageWidth, imageHeight)
            bitmap.recycle()

            val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

            // Step 2: Multi-criteria false positive check
            if (!isRealDetection(keypoints, imageWidth, imageHeight)) {
                emitNotDetected(keypoints, inferenceTimeMs)
                return
            }

            // Step 3: Smooth keypoints to reduce jitter
            val smoothedKeypoints = smoother.smoothKeypoints(keypoints)

            // Step 4: Compute homography from smoothed keypoints
            val rawH = homographyCalculator.computeHomography(smoothedKeypoints)
            if (rawH == null) {
                emitNotDetected(keypoints, inferenceTimeMs)
                return
            }

            // Step 5: Smooth homography for additional stability
            val smoothedH = smoother.smooth(rawH, smoothedKeypoints)

            // Step 6: Project all 16 keypoints
            val allProjected = projector.projectAllKeypoints(smoothedH)

            // Step 7: Validate projected points geometry
            val validProjected = validator.filterValidProjections(
                allProjected, keypoints, imageWidth, imageHeight
            )

            val reprojError = projector.computeReprojectionError(smoothedKeypoints, smoothedH)

            // Step 8: Determine detection status
            val status = when {
                reprojError < 20f && validProjected.size >= 12 -> DetectionStatus.DETECTED
                reprojError < 50f && validProjected.size >= 8 -> DetectionStatus.PARTIAL
                else -> DetectionStatus.NOT_DETECTED
            }

            _detectionResult.value = CourtDetectionResult(
                detectedKeypoints = keypoints,
                projectedKeypoints = validProjected,
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
