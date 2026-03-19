package com.shot.court

import com.shot.core.model.Keypoint

/**
 * Temporal smoothing for detected keypoints using Exponential Moving Average (EMA).
 *
 * Smooths keypoint coordinates across frames to reduce jitter in the court overlay.
 * Also smooths the resulting homography for additional stability.
 * Resets when a sudden scene change is detected (keypoint jump > threshold).
 */
class TemporalSmoother(
    /** EMA smoothing factor for keypoints. Lower = smoother. 0.3 recommended. */
    private val keypointAlpha: Float = 0.3f,

    /** EMA smoothing factor for homography. Lower = smoother. */
    private val homographyAlpha: Float = 0.3f,

    /** Keypoint movement threshold (pixels) to trigger filter reset. */
    private val jumpThreshold: Float = 30f
) {

    private var smoothedKeypoints: MutableMap<Int, FloatArray> = mutableMapOf()
    private var smoothedH: FloatArray? = null
    private var frameCount = 0

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
                val sx = keypointAlpha * kp.x + (1 - keypointAlpha) * prev[0]
                val sy = keypointAlpha * kp.y + (1 - keypointAlpha) * prev[1]
                val sc = keypointAlpha * kp.confidence + (1 - keypointAlpha) * prev[2]
                smoothedKeypoints[kp.id] = floatArrayOf(sx, sy, sc)
                Keypoint(kp.id, sx, sy, sc)
            } else {
                smoothedKeypoints[kp.id] = floatArrayOf(kp.x, kp.y, kp.confidence)
                kp
            }
        }
    }

    /**
     * Smooth homography matrix for additional stability.
     */
    fun smooth(newH: FloatArray, currentKeypoints: List<Keypoint>): FloatArray {
        smoothedH = smoothedH?.let { prev ->
            FloatArray(9) { i -> homographyAlpha * newH[i] + (1 - homographyAlpha) * prev[i] }
        } ?: newH.copyOf()

        return smoothedH!!.copyOf()
    }

    fun reset() {
        smoothedKeypoints.clear()
        smoothedH = null
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
