package com.shot.court

import com.shot.core.model.Keypoint

/**
 * Validates homography matrices using reprojection error.
 */
class HomographyValidator(
    private val projector: CourtProjector = CourtProjector()
) {

    companion object {
        const val MAX_REPROJECTION_ERROR = 50f
    }

    fun isValid(homography: FloatArray, detectedKeypoints: List<Keypoint>): Boolean {
        // Check 1: Determinant is not degenerate
        val det = homography[0] * (homography[4] * homography[8] - homography[5] * homography[7]) -
                homography[1] * (homography[3] * homography[8] - homography[5] * homography[6]) +
                homography[2] * (homography[3] * homography[7] - homography[4] * homography[6])

        if (Math.abs(det) < 1e-10f) return false

        // Check 2: Reprojection error
        val error = projector.computeReprojectionError(detectedKeypoints, homography)
        if (error > MAX_REPROJECTION_ERROR || error.isNaN()) return false

        return true
    }
}
