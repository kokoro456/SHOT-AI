package com.shot.court

import com.shot.core.model.Keypoint

/**
 * Validates homography matrices for geometric consistency and accuracy.
 */
class HomographyValidator(
    private val projector: CourtProjector = CourtProjector()
) {

    companion object {
        /** Maximum allowed mean reprojection error (pixels). */
        const val MAX_REPROJECTION_ERROR = 5f

        /** Maximum allowed distance for projected far-court points from frame (pixels). */
        const val MAX_PROJECTED_DISTANCE = 1000f
    }

    /**
     * Validate a computed homography.
     *
     * Checks:
     * 1. Reprojection error < 5px for detected keypoints
     * 2. Determinant is positive and reasonable
     * 3. Projected far-court keypoints are geometrically valid
     *
     * @param homography 3x3 homography matrix (image → court)
     * @param detectedKeypoints The detected keypoints used to compute H
     * @return true if the homography passes all validation checks
     */
    fun isValid(homography: FloatArray, detectedKeypoints: List<Keypoint>): Boolean {
        // Check 1: Determinant
        if (!isDeterminantValid(homography)) return false

        // Check 2: Reprojection error
        val error = projector.computeReprojectionError(detectedKeypoints, homography)
        if (error > MAX_REPROJECTION_ERROR) return false

        // Check 3: Projected keypoints geometric validity
        val projected = projector.projectAllKeypoints(homography)
        if (!areProjectedPointsValid(projected)) return false

        return true
    }

    /**
     * Check that the homography determinant is positive and within reasonable range.
     * A negative or near-zero determinant indicates a degenerate transform.
     */
    private fun isDeterminantValid(h: FloatArray): Boolean {
        val det = h[0] * (h[4] * h[8] - h[5] * h[7]) -
                h[1] * (h[3] * h[8] - h[5] * h[6]) +
                h[2] * (h[3] * h[7] - h[4] * h[6])
        return det > 1e-6f && det < 1e12f
    }

    /**
     * Check geometric validity of projected points:
     * - Point 1 should be left of point 5 (far baseline)
     * - Far baseline should be above near baseline (smaller y, since camera looks from behind)
     * - All points within reasonable bounds
     */
    private fun areProjectedPointsValid(projected: List<Keypoint>): Boolean {
        if (projected.size < 16) return false

        val byId = projected.associateBy { it.id }

        // Point 1 (far left) should have smaller x than point 5 (far right)
        val p1 = byId[1] ?: return false
        val p5 = byId[5] ?: return false
        if (p1.x >= p5.x) return false

        // Point 12 (near left) should have smaller x than point 16 (near right)
        val p12 = byId[12] ?: return false
        val p16 = byId[16] ?: return false
        if (p12.x >= p16.x) return false

        // Far baseline (points 1-5) should have smaller y than near baseline (12-16)
        // because camera is behind near baseline looking toward far baseline
        val farY = (byId[3]?.y ?: return false)
        val nearY = (byId[14]?.y ?: return false)
        if (farY >= nearY) return false

        // All projected points should be within reasonable bounds
        for (kp in projected) {
            if (Math.abs(kp.x) > MAX_PROJECTED_DISTANCE || Math.abs(kp.y) > MAX_PROJECTED_DISTANCE) {
                return false
            }
        }

        return true
    }
}
