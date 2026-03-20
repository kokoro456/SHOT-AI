package com.shot.court

import com.shot.core.model.Keypoint
import kotlin.math.abs

/**
 * Validates homography matrices and filters geometrically invalid projections.
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

    /**
     * Filter projected keypoints by geometric validity.
     *
     * Near-court points (KP9-16) are always kept (they come from direct detection).
     * Far-court points (KP1-8) are validated against strict geometric constraints:
     * - Must be within tight image bounds (20% margin, not 50%)
     * - Must be above (lower y) the near-court min Y
     * - Far baseline must be narrower than near baseline (perspective foreshortening)
     * - Far-court cannot extend too far above near-court (max 60% of court height)
     *
     * Invalid far-court points are removed to prevent wild overlay rendering.
     */
    fun filterValidProjections(
        projected: List<Keypoint>,
        detected: List<Keypoint>,
        imageWidth: Int,
        imageHeight: Int
    ): List<Keypoint> {
        if (projected.isEmpty()) return projected

        // Separate near-court (KP9-16) and far-court (KP1-8)
        val nearCourt = projected.filter { it.id in 9..16 }
        val farCourt = projected.filter { it.id in 1..8 }

        // Always keep near-court points (from direct detection, reliable)
        val result = nearCourt.toMutableList()

        if (farCourt.isEmpty() || nearCourt.isEmpty()) return result

        // Reference: use MIN y of near-court (topmost point) instead of mean
        val nearYs = detected.filter { it.id in 9..16 }.map { it.y }
        val nearMinY = nearYs.minOrNull() ?: return result
        val nearMaxY = nearYs.maxOrNull() ?: return result
        val nearCourtHeight = nearMaxY - nearMinY

        // Near baseline width (KP12 to KP16)
        val nearLeft = detected.find { it.id == 12 }
        val nearRight = detected.find { it.id == 16 }
        val nearBaselineWidth = if (nearLeft != null && nearRight != null) {
            abs(nearRight.x - nearLeft.x)
        } else {
            imageWidth * 0.5f
        }

        // Far baseline width (KP1 to KP5)
        val farLeft = farCourt.find { it.id == 1 }
        val farRight = farCourt.find { it.id == 5 }
        val farBaselineWidth = if (farLeft != null && farRight != null) {
            abs(farRight.x - farLeft.x)
        } else {
            null
        }

        // Geometric check: far baseline should be narrower than near baseline
        val farNearRatio = if (farBaselineWidth != null && nearBaselineWidth > 0) {
            farBaselineWidth / nearBaselineWidth
        } else {
            null
        }

        // Tighter ratio range: 0.08~0.80 (was 0.05~0.95)
        val farCourtGeometryValid = farNearRatio == null || farNearRatio in 0.08f..0.80f

        // Max allowed vertical extent above near-court top
        val maxFarExtent = if (nearCourtHeight > 0) nearCourtHeight * 1.5f else imageHeight * 0.4f

        // Tighter margin: 20% of image width (was 50%)
        val margin = imageWidth * 0.2f

        // Validate each far-court point
        var validFarCount = 0
        val farCandidates = mutableListOf<Keypoint>()

        for (kp in farCourt) {
            var valid = true

            // Bound check: tighter bounds
            if (kp.x < -margin || kp.x > imageWidth + margin ||
                kp.y < -margin || kp.y > imageHeight + margin) {
                valid = false
            }

            // Far-court points must be above near-court top (use minY, not meanY)
            if (kp.y > nearMinY) {
                valid = false
            }

            // Far-court points cannot extend too far above near-court
            if (kp.y < nearMinY - maxFarExtent) {
                valid = false
            }

            // Overall geometry must be valid
            if (!farCourtGeometryValid) {
                valid = false
            }

            if (valid) {
                farCandidates.add(kp)
                validFarCount++
            }
        }

        // Only add far-court points if majority passed validation (at least 5 of 8)
        // This prevents partial/broken far-court overlays
        if (validFarCount >= 5) {
            result.addAll(farCandidates)
        }

        return result
    }
}
