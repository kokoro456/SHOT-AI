package com.shot.core.model

/**
 * Represents a detected or projected keypoint on the tennis court.
 *
 * @param id Keypoint ID (1-16, matching the court diagram in the spec)
 * @param x X coordinate (pixels in image space)
 * @param y Y coordinate (pixels in image space)
 * @param confidence Detection confidence (0.0-1.0). For projected keypoints, this is 1.0.
 */
data class Keypoint(
    val id: Int,
    val x: Float,
    val y: Float,
    val confidence: Float
) {
    /** Whether this keypoint was reliably detected (confidence above threshold). */
    fun isReliable(threshold: Float = DEFAULT_CONFIDENCE_THRESHOLD): Boolean =
        confidence >= threshold

    companion object {
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.7f

        // Near court keypoint IDs (visible from behind baseline)
        val REQUIRED_IDS = setOf(9, 10, 11, 13, 14, 15)
        val OPTIONAL_IDS = setOf(12, 16)
        val ALL_VISIBLE_IDS = REQUIRED_IDS + OPTIONAL_IDS

        // Far court keypoint IDs (computed via homography)
        val PROJECTED_IDS = setOf(1, 2, 3, 4, 5, 6, 7, 8)

        // All keypoint IDs
        val ALL_IDS = (1..16).toSet()
    }
}

/**
 * Confidence level for visual display.
 */
enum class ConfidenceLevel {
    HIGH,    // > 0.8 - green
    MEDIUM,  // 0.6-0.8 - yellow
    LOW;     // < 0.6 - red

    companion object {
        fun from(confidence: Float): ConfidenceLevel = when {
            confidence > 0.8f -> HIGH
            confidence >= 0.6f -> MEDIUM
            else -> LOW
        }
    }
}
