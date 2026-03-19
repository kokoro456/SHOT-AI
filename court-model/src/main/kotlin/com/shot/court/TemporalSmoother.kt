package com.shot.court

import com.shot.core.model.Keypoint

/**
 * Temporal smoothing for detected keypoints using Exponential Moving Average (EMA)
 * with deadzone suppression and consecutive-frame validation.
 *
 * Smooths keypoint coordinates across frames to reduce jitter in the court overlay.
 * Also smooths the resulting homography for additional stability.
 * Resets when a sudden scene change is detected (keypoint jump > threshold).
 *
 * Anti-jitter strategy:
 *  1. Low EMA alpha (0.35) – heavily weights previous position.
 *  2. Deadzone – movements smaller than [deadzoneThreshold] pixels are ignored,
 *     keeping the overlay perfectly still when the camera is stationary.
 *  3. Consecutive-frame gate – a new detection must appear for
 *     [minConsecutiveFrames] consecutive frames with consistent positions
 *     before it is accepted, filtering out single-frame noise spikes.
 */
class TemporalSmoother(
    /** EMA smoothing factor for keypoints. Lower = smoother. */
    private val keypointAlpha: Float = 0.35f,

    /** EMA smoothing factor for homography. Lower = smoother. */
    private val homographyAlpha: Float = 0.25f,

    /** Keypoint movement threshold (pixels) to trigger filter reset. */
    private val jumpThreshold: Float = 30f,

    /** Movements below this threshold (pixels) are suppressed (deadzone). */
    private val deadzoneThreshold: Float = 4f,

    /** Number of consecutive frames a new keypoint must appear consistently
     *  before it is accepted into the smoothed state. */
    private val minConsecutiveFrames: Int = 3,

    /** Maximum distance (pixels) between consecutive raw detections for them
     *  to be considered "consistent" during the consecutive-frame gate. */
    private val consistencyThreshold: Float = 8f
) {

    private var smoothedKeypoints: MutableMap<Int, FloatArray> = mutableMapOf()
    private var smoothedH: FloatArray? = null
    private var frameCount = 0

    /**
     * Tracks consecutive-frame validation for each keypoint id.
     * Each entry holds [lastRawX, lastRawY, consecutiveCount].
     */
    private val candidateBuffer: MutableMap<Int, FloatArray> = mutableMapOf()

    /**
     * Smooth keypoints before homography computation.
     * Returns smoothed keypoints with stable coordinates.
     */
    fun smoothKeypoints(currentKeypoints: List<Keypoint>): List<Keypoint> {
        if (shouldReset(currentKeypoints)) {
            reset()
        }

        frameCount++

        return currentKeypoints.map { kp ->
            val prev = smoothedKeypoints[kp.id]
            if (prev != null) {
                // ---- Deadzone check ----
                val dx = kp.x - prev[0]
                val dy = kp.y - prev[1]
                val distance = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()

                if (distance < deadzoneThreshold) {
                    // Movement is tiny – keep previous smoothed position exactly.
                    // Still update confidence gently.
                    val sc = keypointAlpha * kp.confidence + (1 - keypointAlpha) * prev[2]
                    smoothedKeypoints[kp.id] = floatArrayOf(prev[0], prev[1], sc)
                    Keypoint(kp.id, prev[0], prev[1], sc)
                } else {
                    // Normal EMA update
                    val sx = keypointAlpha * kp.x + (1 - keypointAlpha) * prev[0]
                    val sy = keypointAlpha * kp.y + (1 - keypointAlpha) * prev[1]
                    val sc = keypointAlpha * kp.confidence + (1 - keypointAlpha) * prev[2]
                    smoothedKeypoints[kp.id] = floatArrayOf(sx, sy, sc)
                    Keypoint(kp.id, sx, sy, sc)
                }
            } else {
                // ---- Consecutive-frame gate for new keypoints ----
                val candidate = candidateBuffer[kp.id]
                if (candidate != null) {
                    val cdx = kp.x - candidate[0]
                    val cdy = kp.y - candidate[1]
                    val cdist = Math.sqrt((cdx * cdx + cdy * cdy).toDouble()).toFloat()

                    if (cdist < consistencyThreshold) {
                        // Consistent with previous candidate – increment counter
                        val count = candidate[2] + 1f
                        candidateBuffer[kp.id] = floatArrayOf(kp.x, kp.y, count)

                        if (count >= minConsecutiveFrames) {
                            // Passed the gate – accept into smoothed state
                            candidateBuffer.remove(kp.id)
                            smoothedKeypoints[kp.id] = floatArrayOf(kp.x, kp.y, kp.confidence)
                            Keypoint(kp.id, kp.x, kp.y, kp.confidence)
                        } else {
                            // Not yet enough frames – return raw but don't commit
                            kp
                        }
                    } else {
                        // Inconsistent – restart candidate tracking
                        candidateBuffer[kp.id] = floatArrayOf(kp.x, kp.y, 1f)
                        kp
                    }
                } else {
                    // First time seeing this keypoint – start candidate tracking
                    candidateBuffer[kp.id] = floatArrayOf(kp.x, kp.y, 1f)
                    kp
                }
            }
        }
    }

    /**
     * Smooth homography matrix for additional stability.
     */
    fun smooth(newH: FloatArray, currentKeypoints: List<Keypoint>): FloatArray {
        smoothedH = smoothedH?.let { prev ->
            // Compute per-element change to apply deadzone on homography too
            var maxDelta = 0f
            for (i in 0 until 9) {
                val delta = Math.abs(newH[i] - prev[i])
                if (delta > maxDelta) maxDelta = delta
            }
            // If the homography barely changed, keep previous exactly
            if (maxDelta < 1e-4f) {
                prev.copyOf()
            } else {
                FloatArray(9) { i -> homographyAlpha * newH[i] + (1 - homographyAlpha) * prev[i] }
            }
        } ?: newH.copyOf()

        return smoothedH!!.copyOf()
    }

    fun reset() {
        smoothedKeypoints.clear()
        smoothedH = null
        candidateBuffer.clear()
        frameCount = 0
    }

    private fun shouldReset(currentKeypoints: List<Keypoint>): Boolean {
        if (smoothedKeypoints.isEmpty()) return false

        var jumpCount = 0
        for (kp in currentKeypoints) {
            val prev = smoothedKeypoints[kp.id] ?: continue
            val dx = kp.x - prev[0]
            val dy = kp.y - prev[1]
            val distance = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
            if (distance > jumpThreshold) jumpCount++
        }

        // Reset only if multiple keypoints jump (not just noise on one point)
        return jumpCount >= 3
    }
}
