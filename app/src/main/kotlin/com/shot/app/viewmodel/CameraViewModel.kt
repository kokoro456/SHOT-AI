package com.shot.app.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.findViewTreeLifecycleOwner
import com.shot.camera.CameraManager
import com.shot.camera.LensDistortionCorrector
import com.shot.court.CourtProjector
import com.shot.court.HomographyCalculator
import com.shot.court.HomographyValidator
import com.shot.court.TemporalSmoother
import com.shot.core.ItfCourtSpec
import com.shot.core.model.CourtDetectionResult
import com.shot.core.model.DetectionStatus
import com.shot.core.model.Keypoint
import com.shot.detection.BallKalmanFilter
import com.shot.detection.BallTrackingDetector
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
) : AndroidViewModel(application), SensorEventListener {

    private val cameraManager = CameraManager(application)
    private var isCameraBound = false

    // ML pipeline components
    private val detector = CourtKeypointDetector(application)
    private val ballDetector = BallTrackingDetector(application)
    private val ballKalmanFilter = BallKalmanFilter()
    private val homographyCalculator = HomographyCalculator()
    private val projector = CourtProjector(homographyCalculator)
    private val validator = HomographyValidator()
    private val smoother = TemporalSmoother()
    private val lensCorrector = LensDistortionCorrector(application, 1280, 720)
    private val shouldUndistort = lensCorrector.hasSignificantDistortion()

    // Ball detection: single-frame model + Kalman filter for temporal tracking

    // Output-level stabilization
    private var lastEmittedProjected: List<Keypoint>? = null
    private var lastEmittedH: FloatArray? = null
    private val outputDeadzoneThreshold = 2.0f

    // Default keypoint positions (model outputs these for non-court scenes)
    private val defaultPositions = floatArrayOf(
        0.29f, 0.53f, 0.52f, 0.53f, 0.66f, 0.55f, 0.29f, 0.71f,
        0.36f, 0.70f, 0.56f, 0.70f, 0.77f, 0.71f, 0.73f, 0.61f
    )

    // --- Court Lock Mode ---
    private var isCourtLocked = false
    private var lockedDetectionResult: CourtDetectionResult? = null

    // --- Ball marker persistence (grace period) ---
    private var lastDetectedBall: BallTrackingDetector.BallPosition? = null
    private var ballMissFrameCount = 0
    private val ballGraceFrames = 4  // Keep showing ball for N frames after losing detection

    // --- G-Sensor movement detection ---
    private val sensorManager = application.getSystemService(SensorManager::class.java)
    private val gyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private var gyroRegistered = false

    // Accumulated angular change since court lock
    private var accumulatedRotation = 0f
    private var lastGyroTimestamp = 0L

    /** Angular change threshold (radians) to trigger movement alert. ~3 degrees */
    private val movementThreshold = 0.05f

    // --- StateFlows ---
    private val _detectionResult = MutableStateFlow(CourtDetectionResult.EMPTY)
    val detectionResult: StateFlow<CourtDetectionResult> = _detectionResult.asStateFlow()

    private val _ballPosition = MutableStateFlow<BallTrackingDetector.BallPosition?>(null)
    val ballPosition: StateFlow<BallTrackingDetector.BallPosition?> = _ballPosition.asStateFlow()

    private val _isDebugMode = MutableStateFlow(false)
    val isDebugMode: StateFlow<Boolean> = _isDebugMode.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _isCourtLockedFlow = MutableStateFlow(false)
    val isCourtLockedFlow: StateFlow<Boolean> = _isCourtLockedFlow.asStateFlow()

    /** True when camera moved significantly after court lock */
    private val _cameraMovedAlert = MutableStateFlow(false)
    val cameraMovedAlert: StateFlow<Boolean> = _cameraMovedAlert.asStateFlow()

    // --- Ball landing spots ---
    data class LandingSpot(
        val imageX: Float, val imageY: Float,
        val courtX: Float, val courtY: Float,
        val isIn: Boolean,
        val timestamp: Long = System.currentTimeMillis()
    )

    private val _landingSpots = MutableStateFlow<List<LandingSpot>>(emptyList())
    val landingSpots: StateFlow<List<LandingSpot>> = _landingSpots.asStateFlow()
    private val maxLandingSpots = 10 // 최근 10개만 유지

    // --- Keypoint adjustment ---
    private val _adjustingKeypointId = MutableStateFlow<Int?>(null)
    val adjustingKeypointId: StateFlow<Int?> = _adjustingKeypointId.asStateFlow()

    private var manualKeypoints: MutableList<Keypoint>? = null

    private val _farCourtPointsAdded = MutableStateFlow(false)
    val farCourtPointsAdded: StateFlow<Boolean> = _farCourtPointsAdded.asStateFlow()

    fun bindCamera(previewView: PreviewView) {
        if (isCameraBound) return
        val lifecycleOwner = previewView.findViewTreeLifecycleOwner() ?: return
        isCameraBound = true
        cameraManager.bind(
            previewView = previewView,
            lifecycleOwner = lifecycleOwner,
            onFrame = { imageProxy -> processFrame(imageProxy) }
        )
    }

    // --- Keypoint Manual Adjustment ---

    /**
     * Update a single keypoint position (user drag).
     * Recomputes homography and reprojects all 16 keypoints.
     */
    fun updateKeypoint(keypointId: Int, imageX: Float, imageY: Float) {
        val current = _detectionResult.value
        if (current.status == DetectionStatus.NOT_DETECTED) return

        // Initialize manual keypoints from current detection
        if (manualKeypoints == null) {
            manualKeypoints = current.detectedKeypoints.toMutableList()
        }

        // Update or add the specific keypoint
        val idx = manualKeypoints!!.indexOfFirst { it.id == keypointId }
        if (idx >= 0) {
            manualKeypoints!![idx] = Keypoint(keypointId, imageX, imageY, 1.0f)
        } else {
            // New keypoint (e.g. far court KP1, KP5 added by user)
            manualKeypoints!!.add(Keypoint(keypointId, imageX, imageY, 1.0f))
        }

        recomputeFromKeypoints()
    }

    /**
     * Add far court doubles corners (KP1 left, KP5 right).
     * Places them at default screen positions for user to drag into place.
     */
    fun addFarCourtPoints() {
        val current = _detectionResult.value
        if (current.status == DetectionStatus.NOT_DETECTED) return

        if (manualKeypoints == null) {
            manualKeypoints = current.detectedKeypoints.toMutableList()
        }

        // Find existing far court keypoints from projected
        val projected = current.projectedKeypoints

        // Use projected positions as initial placement (if available)
        val kp1Projected = projected.find { it.id == 1 }
        val kp5Projected = projected.find { it.id == 5 }

        // Default: top-left and top-right of image if no projection
        val kp1X = kp1Projected?.x ?: 200f
        val kp1Y = kp1Projected?.y ?: 100f
        val kp5X = kp5Projected?.x ?: 1080f
        val kp5Y = kp5Projected?.y ?: 100f

        // Add or update KP1 and KP5
        fun addOrUpdate(id: Int, x: Float, y: Float) {
            val idx = manualKeypoints!!.indexOfFirst { it.id == id }
            if (idx >= 0) {
                manualKeypoints!![idx] = Keypoint(id, x, y, 1.0f)
            } else {
                manualKeypoints!!.add(Keypoint(id, x, y, 1.0f))
            }
        }

        addOrUpdate(1, kp1X, kp1Y)
        addOrUpdate(5, kp5X, kp5Y)

        _farCourtPointsAdded.value = true
        recomputeFromKeypoints()
    }

    private fun recomputeFromKeypoints() {
        val updatedKeypoints = manualKeypoints ?: return
        val newH = homographyCalculator.computeHomography(updatedKeypoints) ?: return
        val allProjected = projector.projectAllKeypoints(newH)
        val validProjected = validator.filterValidProjections(
            allProjected, updatedKeypoints, 1280, 720
        )

        _detectionResult.value = CourtDetectionResult(
            detectedKeypoints = updatedKeypoints,
            projectedKeypoints = validProjected,
            homographyMatrix = newH,
            reprojectionError = projector.computeReprojectionError(updatedKeypoints, newH),
            inferenceTimeMs = 0,
            status = DetectionStatus.DETECTED
        )
    }

    fun setAdjustingKeypoint(id: Int?) {
        _adjustingKeypointId.value = id
    }

    // --- Court Lock Controls ---

    /** Lock court detection at current state. Starts gyroscope monitoring. */
    fun lockCourt() {
        val current = _detectionResult.value
        if (current.status == DetectionStatus.NOT_DETECTED) return

        isCourtLocked = true
        lockedDetectionResult = current
        _isCourtLockedFlow.value = true
        _cameraMovedAlert.value = false

        // Release lens correction LUT (saves ~7MB)
        lensCorrector.release()

        // Start gyroscope monitoring
        accumulatedRotation = 0f
        lastGyroTimestamp = 0L
        if (!gyroRegistered && gyroscope != null) {
            sensorManager?.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_UI)
            gyroRegistered = true
        }
    }

    /** Unlock court detection. Stops gyroscope monitoring. */
    fun unlockCourt() {
        isCourtLocked = false
        lockedDetectionResult = null
        _isCourtLockedFlow.value = false
        _cameraMovedAlert.value = false
        smoother.reset()
        ballKalmanFilter.reset()
        lastEmittedProjected = null
        lastEmittedH = null

        // Stop gyroscope
        if (gyroRegistered) {
            sensorManager?.unregisterListener(this)
            gyroRegistered = false
        }
    }

    /** Dismiss movement alert and re-lock at current position */
    fun dismissMovementAlert() {
        _cameraMovedAlert.value = false
        unlockCourt()
    }

    // --- G-Sensor callbacks ---

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type != Sensor.TYPE_GYROSCOPE) return
        if (!isCourtLocked) return

        val timestamp = event.timestamp
        if (lastGyroTimestamp != 0L) {
            val dt = (timestamp - lastGyroTimestamp) * 1e-9f // nanoseconds → seconds
            // Angular velocity magnitude (rad/s)
            val wx = event.values[0]
            val wy = event.values[1]
            val wz = event.values[2]
            val magnitude = sqrt(wx * wx + wy * wy + wz * wz)
            accumulatedRotation += magnitude * dt

            if (accumulatedRotation > movementThreshold) {
                _cameraMovedAlert.value = true
            }
        }
        lastGyroTimestamp = timestamp
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // --- Image usability check ---

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
        return variance > 100.0
    }

    // --- Real detection check ---

    private fun isRealDetection(keypoints: List<Keypoint>, imageWidth: Int, imageHeight: Int): Boolean {
        if (keypoints.size < 4) return false
        val meanConf = keypoints.map { it.confidence }.average().toFloat()
        if (meanConf < 0.3f) return false
        val confValues = keypoints.map { it.confidence.toDouble() }
        val confMean = confValues.average()
        val confStd = sqrt(confValues.map { (it - confMean) * (it - confMean) }.average()).toFloat()
        if (confStd > 0.35f) return false
        val kp12 = keypoints.find { it.id == 12 }
        val kp16 = keypoints.find { it.id == 16 }
        if (kp12 != null && kp16 != null) {
            val baselineWidth = abs(kp16.x - kp12.x) / imageWidth
            if (baselineWidth < 0.10f) return false
        }
        val defaultDist = distanceFromDefaults(keypoints, imageWidth, imageHeight)
        if (defaultDist < 0.02f) return false
        val reliableCount = keypoints.count { it.confidence > 0.5f }
        if (reliableCount < 4) return false
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
        lastEmittedProjected = null
        lastEmittedH = null
    }

    private fun isOutputStable(prev: List<Keypoint>, current: List<Keypoint>): Boolean {
        if (prev.size != current.size) return false
        val prevById = prev.associateBy { it.id }
        var totalDist = 0f
        var count = 0
        for (kp in current) {
            val p = prevById[kp.id] ?: return false
            val dx = kp.x - p.x
            val dy = kp.y - p.y
            totalDist += sqrt(dx * dx + dy * dy)
            count++
        }
        if (count == 0) return false
        return totalDist / count < outputDeadzoneThreshold
    }

    // --- Ball tracking with Kalman filter ---

    private fun updateBallWithKalman(newBall: BallTrackingDetector.BallPosition) {
        // Step 1: Predict next state
        ballKalmanFilter.predict()

        if (newBall.detected) {
            // Step 2a: Detection succeeded — update Kalman with measurement
            val accepted = ballKalmanFilter.update(newBall.x, newBall.y)
            if (accepted) {
                lastDetectedBall = newBall
                ballMissFrameCount = 0
                _ballPosition.value = newBall
            } else {
                // Measurement rejected (outlier) — use Kalman prediction
                emitKalmanPrediction()
            }
        } else {
            // Step 2b: Detection failed — use Kalman prediction
            ballKalmanFilter.markMiss()
            ballMissFrameCount++
            emitKalmanPrediction()
        }

        // Step 3: Bounce detection (only when court is locked)
        if (isCourtLocked) {
            checkBounce()
        }
    }

    private fun checkBounce() {
        val bouncePos = ballKalmanFilter.detectBounce() ?: return
        val homography = lockedDetectionResult?.homographyMatrix ?: return

        // Convert image coordinates to court coordinates
        val courtCoord = homographyCalculator.imageToCourtCoordinate(
            bouncePos.first, bouncePos.second, homography
        ) ?: return

        val isIn = ItfCourtSpec.isIn(courtCoord.first, courtCoord.second)

        val spot = LandingSpot(
            imageX = bouncePos.first,
            imageY = bouncePos.second,
            courtX = courtCoord.first,
            courtY = courtCoord.second,
            isIn = isIn
        )

        // Add to list, keep only recent N spots
        val current = _landingSpots.value.toMutableList()
        current.add(0, spot)
        if (current.size > maxLandingSpots) {
            current.removeAt(current.size - 1)
        }
        _landingSpots.value = current
    }

    private fun emitKalmanPrediction() {
        val pos = ballKalmanFilter.getPosition()
        if (pos != null) {
            val confidence = ballKalmanFilter.getConfidence()
            _ballPosition.value = BallTrackingDetector.BallPosition(
                x = pos.first,
                y = pos.second,
                confidence = confidence,
                detected = true  // Kalman-predicted position
            )
        } else {
            _ballPosition.value = null
            lastDetectedBall = null
        }
    }

    // --- Main frame processing ---

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val startTime = System.nanoTime()

            val imageWidth = imageProxy.width
            val imageHeight = imageProxy.height

            // --- LOCKED MODE: ball-only fast path (skip bitmap overhead) ---
            if (isCourtLocked) {
                val bitmap = imageProxyToBitmap(imageProxy)
                val ballResult = ballDetector.detect(bitmap, imageWidth, imageHeight)
                updateBallWithKalman(ballResult)
                bitmap.recycle()

                val locked = lockedDetectionResult
                if (locked != null) {
                    _detectionResult.value = locked.copy(
                        inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000
                    )
                }
                return
            }

            // --- UNLOCKED MODE: full pipeline ---
            val rawBitmap = imageProxyToBitmap(imageProxy)

            // Lens distortion correction (unlocked mode only)
            val bitmap = if (shouldUndistort) {
                val undistorted = lensCorrector.undistort(rawBitmap)
                rawBitmap.recycle()
                undistorted
            } else {
                rawBitmap
            }

            // Pre-check: reject uniform/dark images
            if (!isImageUsable(bitmap)) {
                bitmap.recycle()
                emitNotDetected(emptyList(), (System.nanoTime() - startTime) / 1_000_000)
                return
            }

            // Ball detection (every frame)
            val ballResult = ballDetector.detect(bitmap, imageWidth, imageHeight)
            updateBallWithKalman(ballResult)

            // Step 1: Detect keypoints
            val keypoints = detector.detect(bitmap, imageWidth, imageHeight)
            bitmap.recycle()

            val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000

            // Step 2: Multi-criteria false positive check
            if (!isRealDetection(keypoints, imageWidth, imageHeight)) {
                emitNotDetected(keypoints, inferenceTimeMs)
                return
            }

            // Step 3: Smooth keypoints
            val smoothedKeypoints = smoother.smoothKeypoints(keypoints)

            // Step 4: Compute homography
            val rawH = homographyCalculator.computeHomography(smoothedKeypoints)
            if (rawH == null) {
                emitNotDetected(keypoints, inferenceTimeMs)
                return
            }

            // Step 5: Smooth homography
            val smoothedH = smoother.smooth(rawH, smoothedKeypoints)

            // Step 6: Project all 16 keypoints
            val allProjected = projector.projectAllKeypoints(smoothedH)

            // Step 7: Validate projected points
            val validProjected = validator.filterValidProjections(
                allProjected, keypoints, imageWidth, imageHeight
            )

            val reprojError = projector.computeReprojectionError(smoothedKeypoints, smoothedH)

            // Step 8: Determine status
            val status = when {
                reprojError < 25f && validProjected.size >= 10 -> DetectionStatus.DETECTED
                reprojError < 50f && validProjected.size >= 8 -> DetectionStatus.PARTIAL
                else -> DetectionStatus.NOT_DETECTED
            }

            // Step 9: Output stabilization
            val prevProjected = lastEmittedProjected
            val prevH = lastEmittedH
            val useStabilized = prevProjected != null && prevH != null
                    && prevProjected.size == validProjected.size
                    && status != DetectionStatus.NOT_DETECTED
                    && isOutputStable(prevProjected, validProjected)

            val finalProjected = if (useStabilized) prevProjected!! else validProjected
            val finalH = if (useStabilized) prevH!! else smoothedH

            if (!useStabilized) {
                lastEmittedProjected = validProjected
                lastEmittedH = smoothedH.copyOf()
            }

            _detectionResult.value = CourtDetectionResult(
                detectedKeypoints = keypoints,
                projectedKeypoints = finalProjected,
                homographyMatrix = finalH,
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
        if (gyroRegistered) {
            sensorManager?.unregisterListener(this)
        }
        cameraManager.shutdown()
        detector.close()
        ballDetector.close()
    }
}
