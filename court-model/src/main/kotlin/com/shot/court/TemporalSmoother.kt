package com.shot.court

import com.shot.core.model.Keypoint

/**
 * Temporal smoothing for homography matrices using Exponential Moving Average (EMA).
 *
 * Reduces jitter in the court overlay by smoothing consecutive homography estimates.
 * Resets when a sudden scene change is detected (keypoint jump > threshold).
 */
class TemporalSmoother(
    /** EMA smoothing factor. Higher = more weight on new frame. 0.7 = recommended. */
    private val alpha: Float = 0.7f,

    /** Keypoint movement threshold (pixels) to trigger filter reset. */
    private val jumpThreshold: Float = 20f
) {

    private var smoothedH: FloatArray? = null
    private var previousKeypoints: Map<Int, Keypoint>? = null

    /**
     * Smooth a new homography matrix.
     *
     * @param newH Raw homography from current frame (3x3, row-major)
     * @param currentKeypoints Current frame's detected keypoints (for jump detection)
     * @return Smoothed homography matrix
     */
    fun smooth(newH: FloatArray, currentKeypoints: List<Keypoint>): FloatArray {
        // Check for scene change (sudden keypoint jump)
        if (shouldReset(currentKeypoints)) {
            reset()
        }

        smoothedH = smoothedH?.let { prev ->
            FloatArray(9) { i -> alpha * newH[i] + (1 - alpha) * prev[i] }
        } ?: newH.copyOf()

        // Update previous keypoints for next frame's jump detection
        previousKeypoints = currentKeypoints.associateBy { it.id }

        return smoothedH!!.copyOf()
    }

    /**
     * Reset the smoother (e.g., on scene change).
     */
    fun reset() {
        smoothedH = null
        previousKeypoints = null
    }

    /**
     * Detect if keypoints jumped significantly between frames.
     */
    private fun shouldReset(currentKeypoints: List<Keypoint>): Boolean {
        val prev = previousKeypoints ?: return false

        for (kp in currentKeypoints) {
            val prevKp = prev[kp.id] ?: continue
            val dx = kp.x - prevKp.x
            val dy = kp.y - prevKp.y
            val distance = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
            if (distance > jumpThreshold) return true
        }

        return false
    }
}
