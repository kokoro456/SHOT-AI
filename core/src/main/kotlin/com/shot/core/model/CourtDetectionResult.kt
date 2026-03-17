package com.shot.core.model

/**
 * Complete result of court detection for a single frame.
 */
data class CourtDetectionResult(
    /** Keypoints detected by the ML model (6-8 near court points). */
    val detectedKeypoints: List<Keypoint>,

    /** All 16 keypoints after homography projection (detected + computed). */
    val projectedKeypoints: List<Keypoint>,

    /** 3x3 homography matrix (image -> court), flattened row-major. Null if invalid. */
    val homographyMatrix: FloatArray?,

    /** Mean reprojection error in pixels for detected keypoints. */
    val reprojectionError: Float,

    /** ML model inference time in milliseconds. */
    val inferenceTimeMs: Long,

    /** Overall detection status. */
    val status: DetectionStatus,

    /** Frame timestamp in nanoseconds. */
    val timestampNanos: Long = System.nanoTime()
) {
    /** Number of reliably detected keypoints. */
    val reliableKeypointCount: Int
        get() = detectedKeypoints.count { it.isReliable() }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CourtDetectionResult) return false
        return detectedKeypoints == other.detectedKeypoints &&
                projectedKeypoints == other.projectedKeypoints &&
                homographyMatrix.contentEquals(other.homographyMatrix) &&
                reprojectionError == other.reprojectionError &&
                status == other.status
    }

    override fun hashCode(): Int {
        var result = detectedKeypoints.hashCode()
        result = 31 * result + projectedKeypoints.hashCode()
        result = 31 * result + (homographyMatrix?.contentHashCode() ?: 0)
        result = 31 * result + reprojectionError.hashCode()
        result = 31 * result + status.hashCode()
        return result
    }

    companion object {
        /** Empty result for when no court is detected. */
        val EMPTY = CourtDetectionResult(
            detectedKeypoints = emptyList(),
            projectedKeypoints = emptyList(),
            homographyMatrix = null,
            reprojectionError = Float.MAX_VALUE,
            inferenceTimeMs = 0,
            status = DetectionStatus.NOT_DETECTED
        )
    }
}

/**
 * Court detection status.
 */
enum class DetectionStatus {
    /** Court detected with >= 6 reliable keypoints and valid homography. */
    DETECTED,

    /** Some keypoints found but < 6 reliable ones. Court outline may be inaccurate. */
    PARTIAL,

    /** No court found in the frame. */
    NOT_DETECTED
}
